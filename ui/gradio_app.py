import gradio as gr
import requests
import re
import os

# Get API URL from environment variable, fallback to localhost
API = os.getenv("API", "http://127.0.0.1:8000")

_reranker_model = None
def _get_client_reranker():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")
    except Exception:
        _reranker_model = None
    return _reranker_model

def _client_rerank(query, hits, top_k):
    model = _get_client_reranker()
    if model is None:
        return hits
    pairs = [(query, h.get("text","")) for h in hits]
    try:
        scores = model.predict(pairs)
    except Exception:
        return hits
    new_hits = []
    for h, s in zip(hits, list(scores)):
        h2 = dict(h)
        h2["client_rerank_score"] = float(s)
        new_hits.append(h2)
    new_hits.sort(key=lambda x: x.get("client_rerank_score", 0.0), reverse=True)
    return new_hits[:int(top_k)]

def _highlight_snippet(text, query, max_chars=600):
    phrases = re.findall(r"\"([^\"]{3,})\"", query) + re.findall(r"«([^»]{3,})»", query)
    s = text
    for ph in sorted(set(phrases), key=len, reverse=True):
        try:
            s = re.sub(re.escape(ph), lambda m: f"<mark>{m.group(0)}</mark>", s, flags=re.IGNORECASE)
        except re.error:
            pass
    terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3]
    for t in sorted(set(terms), key=len, reverse=True):
        try:
            s = re.sub(rf"(?i)({re.escape(t)})", r"<mark>\1</mark>", s)
        except re.error:
            pass
    snippet = s[:max_chars]
    return snippet

def do_search(q, k, mode, alpha, lexical_fallback, use_client_rerank=False, mmr=False):
    params = {"q": q, "top_k": int(k), "mmr": bool(mmr), "mode": mode, "alpha": float(alpha), "lexical_fallback": bool(lexical_fallback)}
    r = requests.get(f"{API}/search", params=params)
    j = r.json()
    hits = j.get("hits", [])
    if use_client_rerank and mode != "bm25":
        hits = _client_rerank(q, hits, k)
    md = f"### Search Results (mode={mode}, alpha={alpha}, lexical_fallback={lexical_fallback})\n"
    for i, h in enumerate(hits, 1):
        snippet = _highlight_snippet(h.get("text",""), q, max_chars=600)
        extra = f" (client_rerank={h.get('client_rerank_score', 0.0):.3f})" if 'client_rerank_score' in h else ""
        md += f"**{i}. score={h.get('score',0.0):.3f}{extra}** — `{h.get('chunk_id','')}`  \n**{h.get('doc_path','')}** (mode={h.get('mode','')})\n\n> {snippet}\n\n"
    return md

def do_ask(q, k, mode, alpha, lexical_fallback, use_client_rerank=False, mmr=False):
    payload = {"question": q, "top_k": int(k), "mmr": bool(mmr), "mode": mode, "alpha": float(alpha), "lexical_fallback": bool(lexical_fallback)}
    r = requests.post(f"{API}/ask", json=payload)
    j = r.json()
    ans = j.get("answer","")
    hits = j.get("hits", [])
    if use_client_rerank and mode != "bm25":
        hits = _client_rerank(q, hits, k)
    
    # Return answer and sources separately
    answer_md = "### Answer\n" + ans
    
    sources_md = "### Sources with highlighted context\n"
    for i, h in enumerate(hits, 1):
        snippet = _highlight_snippet(h.get("text",""), q, max_chars=600)
        extra = f" (client_rerank={h.get('client_rerank_score', 0.0):.3f})" if 'client_rerank_score' in h else ""
        sources_md += f"**[{i}] {h.get('doc_path','')}** — `{h.get('chunk_id','')}`{extra} (mode={h.get('mode','')})  \n> {snippet}\n\n"
    
    return answer_md, sources_md

with gr.Blocks() as demo:
    gr.Markdown("# 🔎 RAG Semantic Search")
    with gr.Row():
        q = gr.Textbox(label="Your query", placeholder='Example: "Export CSV in Asana" or "Volt-Watt settings"')
    with gr.Row():
        k = gr.Slider(3, 20, value=8, step=1, label="Top-K")
        mode = gr.Dropdown(choices=["hybrid", "vector", "bm25"], value="hybrid", label="Mode")
        alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Alpha (weight of vectors)")
        lexical_fallback = gr.Checkbox(label="Lexical fallback (adaptive alpha)", value=True)
    with gr.Tab("Search"):
        use_rr = gr.Checkbox(label="Client-side re-rank (experimental)", value=False)
        use_mmr = gr.Checkbox(label="Use MMR (diversity) on server", value=False)
        out1 = gr.Markdown()
        gr.Button("Search").click(do_search, [q, k, mode, alpha, lexical_fallback, use_rr, use_mmr], out1)
    with gr.Tab("Ask (RAG)"):
        use_rr2 = gr.Checkbox(label="Client-side re-rank for sources (experimental)", value=False)
        use_mmr2 = gr.Checkbox(label="Use MMR (diversity) on server", value=False)
        
        # Answer output
        answer_out = gr.Markdown(label="Answer")
        
        # Collapsible sources section
        with gr.Accordion("📚 Sources with highlighted context", open=False):
            sources_out = gr.Markdown(label="Sources")
        
        gr.Button("Ask").click(do_ask, [q, k, mode, alpha, lexical_fallback, use_rr2, use_mmr2], [answer_out, sources_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
