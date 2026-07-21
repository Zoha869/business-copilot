from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
import json
from tavily import TavilyClient
import fitz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import re
import networkx as nx
from collections import defaultdict

# Load environment variables
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Initialize Groq client early — the LLM-based triple extractor below needs
# it, and extraction now happens before the FastAPI app is built.
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Read pdfs
pdf_loader = "pdfs"
documents = []

for file in os.listdir(pdf_loader):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_loader, file)
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        documents.append(
            {
                "source": file,
                "text": text
            }
        )
        print(len(documents))
        print(documents[-1]["text"][:300])

# Build Chunks
chunks = []
chunk_size = 500

for doc in documents:
    text = doc["text"]
    for i in range(0, len(text), chunk_size):
        chunks.append({
            "source": doc["source"],
            "text": text[i:i + chunk_size]
        })

print("Total Chunks:", len(chunks))
print(chunks[0])

for index, chunk in enumerate(chunks):
    chunk["id"] = index

print(chunks[0])


# -------- Graph RAG: extraction --------
#
# NOTE: this replaces the old regex-based extract_triples(text), which only
# caught " is " / " has " sentences. The assignment (Zylo W3D2, cell 25)
# specifically asks for LLM triple extraction, so each chunk now gets one
# Groq call that returns (head, relation, tail) triples as JSON.
EXTRACT_PROMPT = """Extract factual relationships from the text as triples.
Each triple is [head_entity, relation, tail_entity]. Use short relation verbs
(owns, offers, requires, targets, causes, part_of, located_in, affects, etc).
Return ONLY a JSON array of triples, nothing else — no markdown fences, no
commentary. Example:
[["bakery", "offers", "custom cakes"], ["custom cakes", "requires", "advance order"]]

Text:
{text}
"""


def llm_extract_triples(text):
    """One LLM call per chunk. Lossy and slower than regex, but this is what
    actually extracts real relationships instead of pattern-matching " is "
    and " has "."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": EXTRACT_PROMPT.format(text=text)}
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        triples = json.loads(raw)
        return [
            (str(t[0]).strip().lower(), str(t[1]).strip().lower(), str(t[2]).strip().lower())
            for t in triples
            if isinstance(t, list) and len(t) == 3
        ]
    except Exception as e:
        print(f"[llm_extract_triples] skipped a chunk: {e}")
        return []


# -------- Graph RAG: entity resolution (the graded part) --------
#
# Collapse alias surface forms ("Payments Team" / "the billing squad") to one
# canonical node. Fill this in after inspecting your own extracted triples —
# without it the graph fractures into disconnected islands.
ALIASES = {
    # Found by inspecting all_triples: singular/plural forms of the same
    # entity were extracted as separate nodes, fracturing the graph.
    "customers": "customer",
    "businesses": "business",
    "existing customers": "existing customer",
}


def canonicalize(entity):
    entity = entity.strip().lower()
    return ALIASES.get(entity, entity)


def build_graph(triples):
    graph = nx.DiGraph()
    for head, relation, tail in triples:
        graph.add_edge(canonicalize(head), canonicalize(tail), relation=relation)
    return graph


def show_fracture(triples, entity_a, entity_b):
    """Proof required by the assignment: build the graph WITHOUT resolution
    and WITH resolution, then use nx.has_path to show a connection that only
    exists once aliases are collapsed. Call this once with a real pair from
    your corpus and paste the printed output into GRAPH.md."""
    graph_raw = nx.DiGraph()
    for head, relation, tail in triples:
        graph_raw.add_edge(head.strip().lower(), tail.strip().lower(), relation=relation)

    graph_resolved = build_graph(triples)

    a_raw, b_raw = entity_a.strip().lower(), entity_b.strip().lower()
    a_res, b_res = canonicalize(entity_a), canonicalize(entity_b)

    raw_connected = (
        a_raw in graph_raw and b_raw in graph_raw and nx.has_path(graph_raw, a_raw, b_raw)
    )
    resolved_connected = (
        a_res in graph_resolved and b_res in graph_resolved
        and nx.has_path(graph_resolved, a_res, b_res)
    )

    print(f"Without resolution → path from '{entity_a}' to '{entity_b}': {raw_connected}")
    print(f"With resolution    → path from '{entity_a}' to '{entity_b}': {resolved_connected}")
    return raw_connected, resolved_connected


# Graph search
def graph_search(query):

    query = query.lower()

    results = []

    for node in G.nodes:

        if query in node or node in query:

            for neighbor in G.neighbors(node):

                relation = G[node][neighbor]["relation"]

                results.append(
                    f"{node} --{relation}--> {neighbor}"
                )

    return "\n".join(results)


# Graph Context
def retrieve_graph_context(query):

    graph_context = graph_search(query)

    if graph_context.strip():
        return graph_context

    return ""


# Graph triple extraction — now LLM-based, one call per chunk
all_triples = []

for chunk in chunks:
    triples = llm_extract_triples(chunk["text"])
    all_triples.extend(triples)

print("Total Triples:", len(all_triples))

for t in all_triples[:10]:
    print(t)

# -------- Build Knowledge Graph (with entity resolution applied) --------

G = build_graph(all_triples)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# TODO (deliverable step): call show_fracture(all_triples, "entity a", "entity b")
# once you've picked a real alias pair from your own corpus, and paste the
# printed before/after into GRAPH.md.


# -------- Graph RAG: finding a multi-hop test question --------
#
# Instead of guessing a question by hand, build a map of which document(s)
# each entity came from, then walk 2-hop chains in the graph and flag any
# chain whose start and end entity never share a source document. Those are
# the candidates worth testing — a single vector-search chunk can't cover
# them, but the graph can walk straight across.
def find_multihop_candidates():
    entity_to_sources = defaultdict(set)

    for chunk in chunks:
        for head, relation, tail in llm_extract_triples(chunk["text"]):
            entity_to_sources[canonicalize(head)].add(chunk["source"])
            entity_to_sources[canonicalize(tail)].add(chunk["source"])

    candidates = []
    for a in G.nodes:
        for b in G.neighbors(a):
            for c in G.neighbors(b):
                if c == a:
                    continue
                if entity_to_sources[a] != entity_to_sources[c]:
                    candidates.append((a, b, c, entity_to_sources[a], entity_to_sources[c]))

    for a, b, c, sources_a, sources_c in candidates:
        print(f"{a} -> {b} -> {c}  |  sources: {sources_a} vs {sources_c}")

    return candidates


# -------- Graph RAG: side-by-side test against vector RAG --------
#
# Runs the same question through both retrievers, prints both contexts, and
# generates an answer from each so you can see exactly where vector RAG
# misses the chain and graph RAG doesn't. This is the comparison the
# deliverable (GRAPH.md) needs — pick a question from
# find_multihop_candidates() above and run it through here.
def compare_rag(question):
    print("QUESTION:", question)
    print("=" * 60)

    vec_context, vec_sources = retrieve_context(question)
    print("VECTOR RAG context:")
    print(vec_context.strip() if vec_context.strip() else "(nothing relevant found)")
    print("-" * 60)

    graph_ctx = retrieve_graph_context(question)
    print("GRAPH RAG context:")
    print(graph_ctx.strip() if graph_ctx.strip() else "(nothing relevant found)")
    print("=" * 60)

    for label, ctx in [("VECTOR", vec_context), ("GRAPH", graph_ctx)]:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Answer only using this context:\n{ctx}"},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=200
        )
        print(f"\n{label} ANSWER:\n{completion.choices[0].message.content}")

# Session with retries — protects against transient connection drops
# (e.g. WinError 10053 / connection aborted) instead of crashing the app.
_jina_session = requests.Session()
_retry_strategy = Retry(
    total=4,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]
)
_jina_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))


# Batched embeddings — sends many chunks of text in ONE request instead of
# one HTTP request per chunk.
def get_embeddings(texts):
    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "jina-embeddings-v3",
        "input": texts
    }

    response = _jina_session.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()

    result_data = sorted(response.json()["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in result_data]


# Kept for compatibility with the single-text retrieval call in /chat.
def get_embedding(text):
    return get_embeddings([text])[0]


embedding = get_embedding(chunks[0]["text"])
print(len(embedding))

# Qdrant client
qdrant_client = QdrantClient(":memory:")

# collection
qdrant_client.create_collection(
    collection_name="business_docs",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

# Load the chunks — embed in batches instead of one request per chunk
points = []
EMBED_BATCH_SIZE = 20

for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
    batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
    batch_texts = [c["text"] for c in batch]
    batch_embeddings = get_embeddings(batch_texts)

    for chunk, chunk_embedding in zip(batch, batch_embeddings):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=chunk_embedding,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"]
                }
            )
        )
    print(f"Embedded {min(batch_start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

qdrant_client.upsert(
    collection_name="business_docs",
    points=points
)
print("Stored Successfully")

# Tavily client
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Model
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.5
    top_p: float = 1.0
    max_tokens: int = 300


# Root Endpoint
@app.get("/")
def root():
    return FileResponse("index.html")

    # If you only want a message instead:
    # return {"message": "The backend is now running. Thanks to Allah."}


# Retrieval helper — embeds the user's message, pulls the top-k matching
# chunks from Qdrant, and also returns the list of unique source filenames
# so the caller can cite them without depending on the LLM to mention them
# on its own.
def retrieve_context(query, k=3, score_threshold=0.35):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="business_docs",
        query=query_embedding,
        limit=k
    ).points

    context = ""
    sources = []

    for result in results:
        if result.score is not None and result.score < score_threshold:
            continue

        source = result.payload["source"]
        if source not in sources:
            sources.append(source)

        context += f"Source: {source}\n"
        context += result.payload["text"] + "\n\n"

    return context, sources


def append_sources(reply_text, sources):
    if not sources:
        return reply_text
    reply_text = (reply_text or "").rstrip()
    return reply_text + "\n\n**Sources:** " + ", ".join(sources)


# Tool Calling
# Sales Data Analyzer
def analyze_sales(sales):
    total_sales = sum(sales)
    average_sales = total_sales / len(sales)
    highest = max(sales)
    lowest = min(sales)

    return {
        "total_sales": total_sales,
        "average_sales": average_sales,
        "highest_sales": highest,
        "lowest_sales": lowest
    }


# Google search tool
def google_search(query):
    response = tavily.search(
        query=query,
        search_depth="basic"
    )
    return response["results"][:3]


# Profit Analyzer
def analyze_profit(revenue, costs):
    profit = revenue - costs
    profit_margin = (profit / revenue * 100) if revenue else 0
    cost_ratio = (costs / revenue * 100) if revenue else 0

    return {
        "revenue": revenue,
        "costs": costs,
        "profit": profit,
        "profit_margin_percent": round(profit_margin, 2),
        "cost_ratio_percent": round(cost_ratio, 2)
    }


# SWOT Analyzer
def analyze_swot(strengths, weaknesses, opportunities, threats):
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "opportunities": opportunities,
        "threats": threats,
        "strengths_count": len(strengths),
        "weaknesses_count": len(weaknesses),
        "opportunities_count": len(opportunities),
        "threats_count": len(threats)
    }


# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message

        if not user_message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        retrieved_context, retrieved_sources = retrieve_context(user_message, k=3)
        graph_context = retrieve_graph_context(user_message)

        system_prompt = """
You are Insight, an AI Business Copilot.

You help small business owners with:
- marketing strategies
- customer growth
- sales improvement
- pricing ideas
- business planning
- competitor analysis

Write like a knowledgeable, friendly consultant talking directly to the
business owner — natural, flowing sentences and short paragraphs, the way
ChatGPT or Claude would answer. Do NOT force every answer into a rigid
template of headings and bullet points. Use **bold** sparingly, just to
highlight a key term. Use bullet or numbered lists only when you are
actually listing multiple distinct steps, options, or items — not as the
default shape of every response.

Keep it concise and avoid filler sentences. If information is missing to
give a precise answer, ask a clarifying question instead of guessing.

If relevant excerpts from the business's own documents are provided below
under "Document Context" or "Graph Context", ground your answer in them and
prefer them over generic knowledge. Graph Context gives you explicit,
connected facts across documents — useful for questions that span more than
one document. If the context is empty or not relevant to the question,
ignore it and just answer normally from your own knowledge.

Only use the analyze_sales, analyze_profit, or analyze_swot tools when the
user actually gives you numbers or lists to analyze. Only use the
google_search tool when the user is clearly asking about something current
— recent news, live prices, this year's trends, or anything you can't
reasonably answer from your own knowledge or the provided document
context. Do NOT use google_search for general definitions, concepts, or
"what is X" questions — answer those directly yourself.

When you do use a tool, weave the result into a natural explanation
instead of dumping raw structured data.
"""

        context_block = ""
        if retrieved_context.strip():
            context_block += "\n\nDocument Context:\n" + retrieved_context
        if graph_context.strip():
            context_block += "\n\nGraph Context (connected facts):\n" + graph_context

        system_prompt_with_context = (
            system_prompt + context_block if context_block else system_prompt
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt_with_context
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_sales",
                    "description": "Analyze business sales data and give insights. Only call this when the user provides actual sales numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sales": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        },
                        "required": ["sales"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": (
                        "Search the web for current, time-sensitive business information "
                        "(recent news, live data, this year's trends). Do NOT call this for "
                        "general definitions, concepts, or explanations you already know."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_profit",
                    "description": "Calculate profit, profit margin, and cost ratio from revenue and costs. Only call this when the user provides actual revenue and cost figures.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "revenue": {
                                "type": "number",
                                "description": "Total revenue earned"
                            },
                            "costs": {
                                "type": "number",
                                "description": "Total costs or expenses"
                            }
                        },
                        "required": ["revenue", "costs"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_swot",
                    "description": "Organize a business's strengths, weaknesses, opportunities, and threats for a SWOT analysis. Only call this when the user actually lists out their strengths/weaknesses/opportunities/threats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strengths": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "weaknesses": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "opportunities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "threats": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["strengths", "weaknesses", "opportunities", "threats"]
                    }
                }
            }
        ]

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            tools=tools
        )

        response_message = completion.choices[0].message

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:

                if tool_call.function.name == "analyze_sales":
                    arguments = json.loads(tool_call.function.arguments)
                    sales_data = arguments["sales"]
                    result = analyze_sales(sales_data)

                    follow_up_messages = messages + [
                        {
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "analyze_sales",
                            "content": json.dumps(result)
                        }
                    ]

                    follow_up = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=follow_up_messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens
                    )

                    final_reply = follow_up.choices[0].message.content

                    return {
                        "status": "success",
                        "tool_used": "analyze_sales",
                        "analysis": result,
                        "reply": final_reply
                    }

                elif tool_call.function.name == "google_search":
                    arguments = json.loads(tool_call.function.arguments)
                    result = google_search(arguments["query"])

                    follow_up_messages = messages + [
                        {
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "google_search",
                            "content": json.dumps(result)
                        }
                    ]

                    follow_up = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=follow_up_messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens
                    )

                    final_reply = follow_up.choices[0].message.content

                    return {
                        "status": "success",
                        "tool_used": "google_search",
                        "search_results": result,
                        "reply": final_reply
                    }

                elif tool_call.function.name == "analyze_profit":
                    arguments = json.loads(tool_call.function.arguments)
                    result = analyze_profit(arguments["revenue"], arguments["costs"])

                    follow_up_messages = messages + [
                        {
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "analyze_profit",
                            "content": json.dumps(result)
                        }
                    ]

                    follow_up = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=follow_up_messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens
                    )

                    final_reply = follow_up.choices[0].message.content

                    return {
                        "status": "success",
                        "tool_used": "analyze_profit",
                        "analysis": result,
                        "reply": final_reply
                    }

                elif tool_call.function.name == "analyze_swot":
                    arguments = json.loads(tool_call.function.arguments)
                    result = analyze_swot(
                        arguments.get("strengths", []),
                        arguments.get("weaknesses", []),
                        arguments.get("opportunities", []),
                        arguments.get("threats", [])
                    )

                    follow_up_messages = messages + [
                        {
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "analyze_swot",
                            "content": json.dumps(result)
                        }
                    ]

                    follow_up = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=follow_up_messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens
                    )

                    final_reply = follow_up.choices[0].message.content

                    return {
                        "status": "success",
                        "tool_used": "analyze_swot",
                        "analysis": result,
                        "reply": final_reply
                    }

            # tool_calls existed but didn't match any known tool
            raise HTTPException(
                status_code=500,
                detail="Model requested an unknown tool."
            )

        # Plain (non-tool) answer — this is the path a question like
        # "what is a business model" takes. Guarantee the document source
        # citation here regardless of whether the model chose to mention it.
        final_reply = append_sources(response_message.content, retrieved_sources)

        return {
            "status": "success",
            "reply": final_reply,
            "sources": retrieved_sources
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------- Manual testing entry point --------
#
# Run this file directly (python main.py) — NOT through uvicorn — to find a
# multi-hop test question and compare vector RAG vs graph RAG on it. Running
# it this way skips starting the API server, so it's just for testing.
if __name__ == "__main__":
    print("\n--- Multi-hop candidates (start -> mid -> end, differing sources) ---")
    find_multihop_candidates()

    compare_rag("How does ingredient quality affect a bakery's sales?")

   
    show_fracture(all_triples, "customer", "riskier spending")