"""
build_index.py
================
Run this ONCE (or whenever your source PDFs change) to build the index:

    python build_index.py

It does all the heavy lifting that used to sit at the top of main.py:
  - load PDFs, chunk them
  - extract (head, relation, tail) triples via the LLM  (parallel, cached)
  - build the knowledge graph + Louvain communities + community reports (parallel, cached)
  - compute embeddings in BATCHES (cached)
  - write everything to disk:
      .rag_cache/qdrant_data/       -> persistent Qdrant collection
      .rag_cache/graph.pkl          -> networkx graph
      .rag_cache/community_reports.json
      .rag_cache/triples_cache.json / reports_cache.json / embeddings_cache.json (unchanged)

main.py then just loads these at startup — no LLM calls, no embedding
computation, no graph building happens when the server boots.
"""

import os
import re
import json
import time
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import fitz
import networkx as nx
import networkx.algorithms.community as nx_community
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GEN_MODEL_NAME = "llama-3.3-70b-versatile"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

EXTRACT_WORKERS = 4
REPORT_WORKERS = 3

CACHE_DIR = ".rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRIPLES_CACHE_PATH = os.path.join(CACHE_DIR, "triples_cache.json")
REPORTS_CACHE_PATH = os.path.join(CACHE_DIR, "reports_cache.json")
EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.json")
GRAPH_PATH = os.path.join(CACHE_DIR, "graph.pkl")
COMMUNITY_REPORTS_PATH = os.path.join(CACHE_DIR, "community_reports.json")
QDRANT_PATH = os.path.join(CACHE_DIR, "qdrant_data")

pdf_loader = "pdfs"
os.makedirs(pdf_loader, exist_ok=True)


def _load_json(p):
    return json.load(open(p, "r")) if os.path.exists(p) else {}


def _save_json(p, d):
    json.dump(d, open(p, "w"))


def _hash_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def gemini_chat(messages, temperature=0, max_tokens=500, retries=3):
    """Kept the name for drop-in compatibility. Calls Groq under the hood."""
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


def main():
    t0 = time.time()
    _triples_cache = _load_json(TRIPLES_CACHE_PATH)
    _reports_cache = _load_json(REPORTS_CACHE_PATH)
    _embed_cache = _load_json(EMBED_CACHE_PATH)

    print("[1/6] Loading + chunking PDFs...")
    documents = []
    for file in os.listdir(pdf_loader):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_loader, file))
            text = "".join(page.get_text() for page in doc)
            documents.append({"source": file, "text": text})

    chunks = []
    chunk_size = 1500
    for doc in documents:
        t = doc["text"]
        for i in range(0, len(t), chunk_size):
            chunks.append({"source": doc["source"], "text": t[i:i + chunk_size], "id": len(chunks)})
    print(f"    {len(documents)} docs -> {len(chunks)} chunks")

    # ---------------------------------------------------------------
    # Triple extraction (parallel, cached, NO artificial sleep needed —
    # gemini_chat already backs off on 429s)
    # ---------------------------------------------------------------
    print("[2/6] Extracting triples...")
    EXTRACT_PROMPT = """Extract triples [head, relation, tail] from the text.
Return ONLY a JSON array. Example: [["apple", "is", "fruit"]]
Text: {text}"""

    def llm_extract_triples(text):
        raw = gemini_chat([{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}])
        if not raw:
            return []
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw)
            return [(str(t[0]).lower(), str(t[1]).lower(), str(t[2]).lower()) for t in data if len(t) == 3]
        except Exception:
            return []

    all_triples = []
    to_extract = []
    for c in chunks:
        key = _hash_text(c["text"])
        if key in _triples_cache:
            all_triples.extend(_triples_cache[key])
        else:
            to_extract.append((key, c))

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
        future_to_key = {executor.submit(llm_extract_triples, c["text"]): key for key, c in to_extract}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            res = future.result()
            _triples_cache[key] = res
            all_triples.extend(res)

    _save_json(TRIPLES_CACHE_PATH, _triples_cache)

    G = nx.DiGraph()
    for h, r, t in all_triples:
        G.add_edge(h.strip(), t.strip(), relation=r.strip())

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

    # ---------------------------------------------------------------
    # Communities + reports (parallel now, was sequential + sleep(1))
    # ---------------------------------------------------------------
    print("[3/6] Detecting communities + generating reports...")
    try:
        communities = nx_community.louvain_communities(G.to_undirected(), seed=42)
        communities = [c for c in communities if len(c) > 2][:10]
    except Exception:
        communities = []

    REPORT_PROMPT = "Summarize these facts for a business owner:\n{facts}"

    def get_community_report(nodes):
        facts = ""
        for h, t, d in G.subgraph(nodes).edges(data=True):
            facts += f"- {h} {d['relation']} {t}\n"
        return gemini_chat([{"role": "user", "content": REPORT_PROMPT.format(facts=facts)}])

    community_reports = []
    to_report = []
    for comm in communities:
        key = _hash_text(",".join(sorted(comm)))
        if key in _reports_cache:
            community_reports.append(_reports_cache[key])
        else:
            to_report.append((key, comm))

    with ThreadPoolExecutor(max_workers=REPORT_WORKERS) as executor:
        future_to_key = {executor.submit(get_community_report, comm): key for key, comm in to_report}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            rep = future.result()
            if rep:
                _reports_cache[key] = rep
                community_reports.append(rep)

    _save_json(REPORTS_CACHE_PATH, _reports_cache)
    _save_json(COMMUNITY_REPORTS_PATH, community_reports)

    # ---------------------------------------------------------------
    # Embeddings — BATCHED, and the append-to-points bug is fixed here
    # ---------------------------------------------------------------
    print("[4/6] Computing embeddings (batched)...")
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    def get_embeddings_batch(texts: List[str]):
        try:
            embeddings = _embed_model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            print(f"[embeddings] Error: {e}")
            return [([0.0] * EMBED_DIM) for _ in texts]

    points = []
    to_embed_idx = []
    to_embed_texts = []

    for c in chunks:
        key = _hash_text(c["text"])
        cached = _embed_cache.get(key)
        if cached is not None and len(cached) == EMBED_DIM:
            points.append(PointStruct(id=c["id"], vector=cached, payload=c))
        else:
            to_embed_idx.append((key, c))
            to_embed_texts.append(c["text"])

    if to_embed_texts:
        new_embs = get_embeddings_batch(to_embed_texts)
        for (key, c), emb in zip(to_embed_idx, new_embs):
            _embed_cache[key] = emb
            points.append(PointStruct(id=c["id"], vector=emb, payload=c))

    _save_json(EMBED_CACHE_PATH, _embed_cache)
    print(f"    {len(points)} chunks embedded (of which {len(to_embed_texts)} newly computed)")

    # ---------------------------------------------------------------
    # Persistent Qdrant (on disk, not ":memory:") so main.py can just load it
    # ---------------------------------------------------------------
    print("[5/6] Writing persistent vector index...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    if qdrant_client.collection_exists("business_docs"):
        qdrant_client.delete_collection("business_docs")
    qdrant_client.create_collection(
        collection_name="business_docs",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    if points:
        qdrant_client.upsert("business_docs", points)
    qdrant_client.close()

    print(f"[6/6] Done in {time.time() - t0:.1f}s. Index is in ./{CACHE_DIR} — main.py will load it at startup.")


if __name__ == "__main__":
    main()