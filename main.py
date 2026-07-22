import os
import time
import json
import re
import hashlib
from collections import defaultdict, deque
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI & Security
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt as pyjwt
from jwt import PyJWKClient
from supabase import create_client, Client
from dotenv import load_dotenv

# Search & Graph
from groq import Groq
from tavily import TavilyClient
import fitz
import networkx as nx
import networkx.algorithms.community as nx_community
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# NOTE: Groq does not expose an embeddings endpoint, so embeddings are produced
# locally with sentence-transformers instead of a remote embedding API.
from sentence_transformers import SentenceTransformer

# ============================================================
# Env / Configuration
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- was GEMINI_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

groq_client = Groq(api_key=GROQ_API_KEY)

# Fast Groq-hosted model for chat/extraction
GEN_MODEL_NAME = "llama-3.3-70b-versatile"

# Local embedding model (replaces Gemini's text-embedding-004, dim=768 -> 384)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Rate limit / worker settings — Groq's free tier is generous vs Gemini's 15 RPM,
# but we keep conservative defaults; raise these if your Groq plan allows more.
EXTRACT_WORKERS = 4
EMBED_WORKERS = 4
REPORT_WORKERS = 3

# ============================================================
# Groq Wrapper Functions
# ============================================================
def gemini_chat(messages, temperature=0, max_tokens=500, retries=3):
    """
    Kept the original function name for drop-in compatibility with the rest
    of the file, but this now calls Groq's chat completions API.
    """
    prompt = messages[-1]["content"]
    delay = 2.0

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GEN_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(delay)
                delay *= 2
                continue
            print(f"[gemini_chat] Error: {e}")
            return None
    return None

def get_embeddings_batch(texts: List[str]):
    """Local sentence-transformers batch embeddings."""
    try:
        embeddings = _embed_model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"[embeddings] Error: {e}")
        return [([0.0] * EMBED_DIM) for _ in texts]

def get_embedding(text: str):
    return get_embeddings_batch([text])[0]

# ============================================================
# Storage & Cache Initialization
# ============================================================
CACHE_DIR = ".rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRIPLES_CACHE_PATH = os.path.join(CACHE_DIR, "triples_cache.json")
REPORTS_CACHE_PATH = os.path.join(CACHE_DIR, "reports_cache.json")
EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.json")

def _load_json(p): return json.load(open(p, "r")) if os.path.exists(p) else {}
def _save_json(p, d): json.dump(d, open(p, "w"))

_triples_cache = _load_json(TRIPLES_CACHE_PATH)
_reports_cache = _load_json(REPORTS_CACHE_PATH)
_embed_cache = _load_json(EMBED_CACHE_PATH)

def _hash_text(t): return hashlib.sha256(t.encode()).hexdigest()

# ============================================================
# Document Processing
# ============================================================
pdf_loader, excel_loader = "pdfs", "excel_files"
os.makedirs(pdf_loader, exist_ok=True)
os.makedirs(excel_loader, exist_ok=True)

documents = []
for file in os.listdir(pdf_loader):
    if file.endswith(".pdf"):
        doc = fitz.open(os.path.join(pdf_loader, file))
        text = "".join(page.get_text() for page in doc)
        documents.append({"source": file, "text": text})

# Chunks
chunks = []
chunk_size = 1500
for doc in documents:
    t = doc["text"]
    for i in range(0, len(t), chunk_size):
        chunks.append({"source": doc["source"], "text": t[i:i+chunk_size], "id": len(chunks)})

# ============================================================
# Graph RAG Extraction
# ============================================================
EXTRACT_PROMPT = """Extract triples [head, relation, tail] from the text. 
Return ONLY a JSON array. Example: [["apple", "is", "fruit"]]
Text: {text}"""

def llm_extract_triples(text):
    raw = gemini_chat([{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}])
    if not raw: return []
    # Clean markdown
    raw = re.sub(r"json|", "", raw).strip()
    try:
        data = json.loads(raw)
        return [(str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()) for t in data if len(t) == 3]
    except: return []

# Build Graph
all_triples = []
with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
    futures = []
    for c in chunks:
        key = _hash_text(c["text"])
        if key in _triples_cache:
            all_triples.extend(_triples_cache[key])
        else:
            futures.append(executor.submit(llm_extract_triples, c["text"]))
            time.sleep(0.3)  # Groq's limits are looser than Gemini's free tier
    for f in as_completed(futures):
        res = f.result()
        all_triples.extend(res)

G = nx.DiGraph()
for h, r, t in all_triples:
    G.add_edge(h.strip(), t.strip(), relation=r.strip())

# ============================================================
# Community Summaries (Global Context)
# ============================================================
try:
    communities = nx_community.louvain_communities(G.to_undirected(), seed=42)
    communities = [c for c in communities if len(c) > 2][:10]
except:
    communities = []

community_reports = []
REPORT_PROMPT = "Summarize these facts for a business owner:\n{facts}"

def get_community_report(nodes):
    facts = ""
    for h, t, d in G.subgraph(nodes).edges(data=True):
        facts += f"- {h} {d['relation']} {t}\n"
    return gemini_chat([{"role": "user", "content": REPORT_PROMPT.format(facts=facts)}])

for comm in communities:
    key = _hash_text(",".join(sorted(comm)))
    if key in _reports_cache:
        community_reports.append(_reports_cache[key])
    else:
        rep = get_community_report(comm)
        if rep:
            _reports_cache[key] = rep
            community_reports.append(rep)
        time.sleep(1)

_save_json(REPORTS_CACHE_PATH, _reports_cache)

# ============================================================
# Vector DB (Qdrant)
# ============================================================
qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="business_docs",
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),  # 384 for MiniLM
)

points = []
for c in chunks:
    key = _hash_text(c["text"])
    if key in _embed_cache:
        emb = _embed_cache[key]
    else:
        emb = get_embedding(c["text"])
        _embed_cache[key] = emb
    points.append(PointStruct(id=c["id"], vector=emb, payload=c))

if points:
    qdrant_client.upsert("business_docs", points)
_save_json(EMBED_CACHE_PATH, _embed_cache)

# ============================================================
# Tools Definition (Groq function-calling schema)
# ============================================================
def google_search(query: str):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily.search(query=query, search_depth="basic")["results"][:3]

def analyze_sales(sales: list):
    return {"total": sum(sales), "avg": sum(sales)/len(sales), "max": max(sales)}

# Groq's API is OpenAI-compatible: tools must be JSON-schema function defs,
# not raw Python callables like Gemini's SDK accepted.
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
                "properties": {
                    "sales": {"type": "array", "items": {"type": "number"}}
                },
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Auth: verify Supabase JWT from the Authorization header
# ============================================================
async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]

    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server auth is not configured")

    try:
        payload = pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

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
# Basic prompt-injection flag (best-effort heuristic, not a hard guarantee)
# ============================================================
_INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"disregard (all|any|previous) instructions",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions)",
]

def flag_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in _INJECTION_PATTERNS)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    check_rate_limit(user_id)

    # 1. Retrieval
    query_emb = get_embedding(request.message)
    search_res = qdrant_client.query_points("business_docs", query_emb, limit=3).points
    context = "\n".join([r.payload["text"] for r in search_res])
    sources = list(set([r.payload["source"] for r in search_res]))

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

        # Send tool results back for the final answer
        response = groq_client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=messages,
        )
        msg = response.choices[0].message

    final_text = msg.content
    if sources:
        final_text += f"\n\n*Sources:* {', '.join(sources)}"

    return {"reply": final_text, "status": "success"}

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    check_rate_limit(user_id)

    async def event_generator():
        stream = groq_client.chat.completions.create(
            model=GEN_MODEL_NAME,
            messages=[{"role": "user", "content": request.message}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ============================================================
# Supabase / DB Helpers (Simplified for snippet)
# ============================================================
supabase_admin = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# (Add your list_conversations, get_messages endpoints here...)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)