import os
import time
import json
import pickle
import uuid
from collections import defaultdict, deque
from typing import Optional

# FastAPI & Security
from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt as pyjwt
from jwt import PyJWKClient
from supabase import create_client, Client
from dotenv import load_dotenv

import fitz  # PyMuPDF, for reading uploaded PDFs
from groq import Groq
from tavily import TavilyClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# ============================================================
# Env / Configuration
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

groq_client = Groq(api_key=GROQ_API_KEY)
GEN_MODEL_NAME = "llama-3.3-70b-versatile"

# Supabase now signs auth tokens with an asymmetric key by default (ES256/RS256),
# verified via its public JWKS endpoint. We try that first, and fall back to the
# older shared-secret HS256 verification for projects still on the legacy setup.
_jwks_client = None
if SUPABASE_URL:
    try:
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    except Exception as e:
        print(f"[auth] Could not set up JWKS client: {e}")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

CACHE_DIR = ".rag_cache"
GRAPH_PATH = os.path.join(CACHE_DIR, "graph.pkl")
COMMUNITY_REPORTS_PATH = os.path.join(CACHE_DIR, "community_reports.json")
QDRANT_PATH = os.path.join(CACHE_DIR, "qdrant_data")

# ============================================================
# Startup: LOAD the pre-built index — no LLM calls, no embedding
# computation, no graph building happens here. Run build_index.py
# separately whenever your source PDFs change.
# ============================================================
if not os.path.exists(QDRANT_PATH):
    raise RuntimeError(
        f"No index found at {QDRANT_PATH}. Run `python build_index.py` first to build it."
    )

print("Loading embedding model...")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading vector index...")
qdrant_client = QdrantClient(path=QDRANT_PATH)

print("Loading graph + community reports...")
G = None
if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

community_reports = []
if os.path.exists(COMMUNITY_REPORTS_PATH):
    with open(COMMUNITY_REPORTS_PATH, "r") as f:
        community_reports = json.load(f)

print("Ready.")


def get_embedding(text: str):
    return _embed_model.encode([text], convert_to_numpy=True)[0].tolist()


def retrieve_context(query: str, limit: int = 3):
    """Shared retrieval step used by both /chat and /chat/stream so both
    endpoints answer questions about uploaded PDFs consistently."""
    query_emb = get_embedding(query)
    search_res = qdrant_client.query_points("business_docs", query_emb, limit=limit).points
    context = "\n".join([r.payload["text"] for r in search_res])
    sources = list(set([r.payload["source"] for r in search_res]))
    return context, sources


# ============================================================
# Tools Definition (Groq function-calling schema)
# ============================================================
def google_search(query: str):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily.search(query=query, search_depth="basic")["results"][:3]


def analyze_sales(sales: list):
    return {"total": sum(sales), "avg": sum(sales) / len(sales), "max": max(sales)}


tools_list = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales",
            "description": "Compute total, average, and max of a list of sales figures",
            "parameters": {
                "type": "object",
                "properties": {"sales": {"type": "array", "items": {"type": "number"}}},
                "required": ["sales"],
            },
        },
    },
]

TOOL_IMPLS = {"google_search": google_search, "analyze_sales": analyze_sales}

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ConversationCreate(BaseModel):
    title: str = "New chat"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/")
async def root():
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {
        "status": "ok",
        "message": "Business Copilot API is running, but index.html was not found next to main.py.",
        "docs": "Go to /docs to try the /chat endpoint from the browser.",
    }


# ============================================================
# Auth: verify Supabase JWT from the Authorization header
# ============================================================
async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]

    payload = None
    last_error = None

    # 1. Try Supabase's current asymmetric-key (JWKS) verification
    if _jwks_client:
        try:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                audience="authenticated",
            )
        except Exception as e:
            last_error = e

    # 2. Fall back to the legacy shared-secret HS256 verification
    if payload is None and SUPABASE_JWT_SECRET:
        try:
            payload = pyjwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except Exception as e:
            last_error = e

    if payload is None:
        detail = "Invalid token"
        if last_error is not None:
            detail = f"Invalid token: {last_error}"
        raise HTTPException(status_code=401, detail=detail)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim")

    return user_id


# ============================================================
# Simple in-memory sliding-window rate limiter (per user_id)
# ============================================================
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log = defaultdict(deque)


def check_rate_limit(user_id: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    log = _request_log[user_id]

    while log and log[0] < window_start:
        log.popleft()

    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")

    log.append(now)


# ============================================================
# Supabase / DB Helpers
# ============================================================
supabase_admin: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def require_db():
    if not supabase_admin:
        raise HTTPException(status_code=500, detail="Supabase is not configured on the server")


def get_owned_conversation(conversation_id: str, user_id: str):
    """Fetch a conversation only if it belongs to this user, else 404."""
    res = (
        supabase_admin.table("conversations")
        .select("id, title")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return res.data[0]


def save_message(conversation_id: str, role: str, content: str):
    supabase_admin.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }).execute()


def touch_conversation(conversation_id: str):
    supabase_admin.table("conversations").update({
        "updated_at": "now()"
    }).eq("id", conversation_id).execute()


def generate_title(first_message: str) -> str:
    """Ask the LLM for a short (3-6 word) title, with a plain-text fallback."""
    try:
        res = groq_client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the user's message into a short chat title, "
                        "3 to 6 words, no quotes, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": first_message},
            ],
            max_tokens=20,
        )
        title = res.choices[0].message.content.strip().strip('"')
        return title[:80] if title else first_message[:40]
    except Exception:
        return first_message[:40]


def maybe_update_title(conversation_id: str, user_message: str) -> Optional[str]:
    """If this conversation only has the messages we just wrote for this turn
    (i.e. it's the first exchange), generate and store a real title."""
    count_res = (
        supabase_admin.table("messages")
        .select("id", count="exact")
        .eq("conversation_id", conversation_id)
        .execute()
    )
    if count_res.count is not None and count_res.count <= 2:
        title = generate_title(user_message)
        supabase_admin.table("conversations").update({"title": title}).eq(
            "id", conversation_id
        ).execute()
        return title
    return None


# ============================================================
# Conversations API
# ============================================================
@app.post("/conversations")
async def create_conversation(
    body: ConversationCreate, user_id: str = Depends(get_current_user)
):
    require_db()
    res = supabase_admin.table("conversations").insert({
        "user_id": user_id,
        "title": body.title,
    }).execute()
    return {"conversation": res.data[0]}


@app.get("/conversations")
async def list_conversations(user_id: str = Depends(get_current_user)):
    require_db()
    res = (
        supabase_admin.table("conversations")
        .select("id, title, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return {"conversations": res.data}


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str, user_id: str = Depends(get_current_user)
):
    require_db()
    get_owned_conversation(conversation_id, user_id)

    res = (
        supabase_admin.table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )
    return {"messages": res.data}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, user_id: str = Depends(get_current_user)
):
    require_db()
    get_owned_conversation(conversation_id, user_id)
    supabase_admin.table("conversations").delete().eq("id", conversation_id).execute()
    return {"status": "deleted"}


# ============================================================
# Chat endpoints
# ============================================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    check_rate_limit(user_id)

    # 1. Retrieval
    context, sources = retrieve_context(request.message)

    # 2. Build Groq chat messages
    messages = [
        {"role": "system", "content": f"You are Insight, an AI Business Copilot. Context: {context}"},
        {"role": "user", "content": request.message},
    ]

    # 3. Generate (with tool-calling)
    response = groq_client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=messages,
        tools=tools_list,
        tool_choice="auto",
    )
    msg = response.choices[0].message

    # 4. Handle Tool Calls (Groq/OpenAI style)
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            if name in TOOL_IMPLS:
                res = TOOL_IMPLS[name](**args)
            else:
                res = "Error: Tool not found"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(res),
            })

        response = groq_client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=messages,
        )
        msg = response.choices[0].message

    final_text = msg.content
    if sources:
        final_text += f"\n\n*Sources:* {', '.join(sources)}"

    if request.conversation_id and supabase_admin:
        get_owned_conversation(request.conversation_id, user_id)
        save_message(request.conversation_id, "user", request.message)
        save_message(request.conversation_id, "assistant", final_text)
        maybe_update_title(request.conversation_id, request.message)
        touch_conversation(request.conversation_id)

    return {"reply": final_text, "status": "success"}


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    check_rate_limit(user_id)

    conversation_id = request.conversation_id
    if conversation_id and supabase_admin:
        get_owned_conversation(conversation_id, user_id)
        save_message(conversation_id, "user", request.message)

    # Same retrieval step as /chat, so streamed answers can also see
    # whatever PDFs were uploaded/indexed.
    context, sources = retrieve_context(request.message)

    async def event_generator():
        accumulated = ""
        try:
            stream = groq_client.chat.completions.create(
                model=GEN_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Insight, an AI Business Copilot. Context: {context}",
                    },
                    {"role": "user", "content": request.message},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # Append sources once the reply is fully streamed, same as /chat does.
        if sources:
            sources_note = f"\n\n*Sources:* {', '.join(sources)}"
            accumulated += sources_note
            yield f"data: {json.dumps({'type': 'token', 'content': sources_note})}\n\n"

        title = None
        if conversation_id and supabase_admin:
            if accumulated.strip():
                save_message(conversation_id, "assistant", accumulated)
            title = maybe_update_title(conversation_id, request.message)
            touch_conversation(conversation_id)

        done_payload = {"type": "done"}
        if title:
            done_payload["title"] = title
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)
UPLOAD_CHUNK_SIZE = 1500
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    check_rate_limit(user_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File is too large (max 20 MB).")

    # Save a copy under pdfs/ so it's picked up if build_index.py is ever
    # re-run from scratch later.
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(PDF_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(raw)

    # Extract text and chunk it the same way build_index.py does.
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't read that PDF — is it a valid file?")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in that PDF.")

    chunk_texts = [text[i:i + UPLOAD_CHUNK_SIZE] for i in range(0, len(text), UPLOAD_CHUNK_SIZE)]

    # Batch-embed all new chunks in one go and upsert them straight into the
    # already-loaded, persistent Qdrant collection — no need to rerun the
    # whole build_index.py pipeline just to add one document.
    embeddings = _embed_model.encode(chunk_texts, convert_to_numpy=True)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"source": safe_name, "text": chunk_text},
        )
        for chunk_text, emb in zip(chunk_texts, embeddings)
    ]
    qdrant_client.upsert("business_docs", points)

    return {
        "status": "success",
        "filename": safe_name,
        "chunks_indexed": len(points),
        "message": "Document uploaded and indexed. You can ask questions about it right away.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)