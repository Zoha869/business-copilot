# Graph RAG — Business Copilot Knowledge Graph

**Project:** AI Business Copilot (Insight)
**Corpus:** `01_Business_Strategy_Guide.pdf`, `02_Sales_Finance_Handbook.pdf`, `03_Marketing_Customer_Growth_Guide.pdf`

---

## 1. Extraction schema

Triples are extracted with one LLM call per chunk (Groq `llama-3.1-8b-instant`, temperature 0), not with a fixed-type regex parser. Each chunk is asked to return `(head, relation, tail)` as a JSON array.

**Vocabulary choice: open, not fixed.** The prompt suggests short verbs (`owns`, `offers`, `requires`, `targets`, `causes`, `part_of`, `located_in`, `affects`) but does not restrict the model to only those. This was a deliberate trade-off:

- **Fixed types** would be precise and easy to query, but this corpus spans four unrelated business domains (bakery, salon, restaurant, gym, SaaS) — a fixed schema would either miss most domain-specific relations or need constant expansion.
- **Open vocabulary** captures far more of what the documents actually say, at the cost of noisier / less consistent relation labels (e.g. `is`, `has`, `offers`, `targets`, `affects` all appear for what is sometimes the same kind of relationship).

Given the copilot has to serve arbitrary small businesses, open extraction was the right trade-off for coverage over precision.

---

## 2. Entity resolution

Inspecting the extracted triples surfaced a real fracture pattern: several entities were extracted in **both singular and plural surface forms** as separate graph nodes:

| Split forms | Canonical |
|---|---|
| `customer` / `customers` | `customer` |
| `business` / `businesses` | `business` |
| `existing customer` / `existing customers` | `existing customer` |

This is exactly the kind of alias fracture the assignment describes — the LLM extractor has no memory across chunks, so the same real-world entity gets a different string depending on how a given sentence phrased it. The `ALIASES` dict + `canonicalize()` collapse these to one node each before the graph is built.

---

## 3. Fracture proof — and the honest failure

**Test:** `show_fracture(all_triples, "customer", "riskier spending")`

```
Without resolution → path from 'customer' to 'riskier spending': True
With resolution    → path from 'customer' to 'riskier spending': True
```

**This did not prove what it was supposed to.** The expectation was `False → True` (no path without resolution, a path appears once `customers` merges into `customer`). Instead both came back `True`.

**Why:** the raw graph already had a direct route that had nothing to do with the customer/customers split — `reviews -> customer -> growth` and, separately, `repeat customers -> growth -> riskier spending` both create an edge `growth -> riskier spending`. Since the *singular* node `customer` already had its own edge straight to `growth` (via the `reviews` triple), `customer -> growth -> riskier spending` was reachable with zero resolution needed. The alias merge was real, but this particular entity pair happened to already be connected through an unrelated path, so it wasn't a valid witness for the fracture.

**Lesson / what I'd do differently:** before picking a fracture pair, check `nx.has_path` on the *raw* graph for candidate pairs first and only use one that comes back `False` — otherwise the alias merge isn't actually doing the work you're trying to demonstrate. A cleaner witness would isolate a tail node that is *only* reachable through the plural form (e.g. a tail that appears solely under `customers -> X -> Y` and nowhere under any `customer -> ...` edge), rather than a hub entity like `customer`/`business` that ends up connected to almost everything through some other route.

---

## 4. Beating vector RAG — the winning multi-hop query

**Chain found by `find_multihop_candidates()`:**
```
ingredient quality -> repeat business -> sales
  sources: {02_Sales_Finance_Handbook.pdf} vs {03_Marketing_Customer_Growth_Guide.pdf}
```
Start and end entity come from two different documents — a single vector chunk can't contain the full chain.

**Question:** *"How does ingredient quality affect a bakery's sales?"*

**Vector RAG answer** (top-k chunk similarity):
> Ingredient quality is one of the factors that can affect a bakery's sales... a home bakery's cakes use better ingredients, which justifies value-based pricing... This suggests that customers are willing to pay more for high-quality ingredients, which can positively impact the bakery's sales.

This answer is a *plausible-sounding guess built from a pricing-strategy chunk* — it never actually retrieved the sentence that states the mechanism. It leans on inference ("suggests") rather than a stated fact.

**Graph RAG answer** (subgraph traversal):
> Ingredient quality affects a bakery's sales by influencing customer satisfaction and loyalty. High-quality ingredients can result in consistent and delicious products, which in turn can lead to repeat business and positive word-of-mouth. This can increase a bakery's sales by attracting and retaining customers.

The graph context handed the model an **explicit edge**: `ingredient quality --affects--> repeat business`, extracted directly from the Sales & Finance Handbook, plus the surrounding bakery subgraph. The graph answer states the mechanism (ingredient quality → repeat business → sales) as a fact rather than inferring it — because the fact was retrieved, not guessed.

**Why vector RAG missed it:** the sentence connecting ingredient quality to repeat business lives in `02_Sales_Finance_Handbook.pdf`, in a paragraph actually about *cost-cutting mistakes* — a query about "ingredient quality and sales" doesn't score highly against that paragraph by embedding similarity, since the paragraph's dominant topic is cash flow and expense-cutting, not sales growth. The graph traversal doesn't care what a chunk is "mostly about" — it just follows the one edge that matters.

---

## 5. Summary

| Step | Status |
|---|---|
| LLM triple extraction | Done — replaced regex extractor with Groq-based extraction |
| Entity resolution | Done — `customer`/`customers`, `business`/`businesses` aliases found and merged |
| Fracture proof | Attempted, did not isolate a clean witness — documented as the honest failure (§3) |
| Multi-hop traversal | Done — 2-hop `graph_search` with anchor + BFS walk |
| Beat vector RAG | Done — "ingredient quality → repeat business → sales" query, graph answer states the mechanism directly, vector answer infers it |