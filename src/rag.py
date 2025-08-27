import os
from typing import List, Dict
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Init OpenAI client if key is present
_api_key = os.getenv("OPENAI_API_KEY")
if _api_key and _api_key.strip() and _api_key != "your_openai_api_key_here":
    client = OpenAI(api_key=_api_key)
    OPENAI_AVAILABLE = True
else:
    client = None
    OPENAI_AVAILABLE = False

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

def _format_source(block: Dict, idx: int) -> str:
    doc = block.get("doc_path", "unknown")
    cid = block.get("chunk_id", "chunk")
    return f"[{idx}] {doc} — `{cid}`"

def build_context(blocks: List[Dict], max_chars: int = 3500) -> str:
    """Join retrieved blocks into a single context string with soft cap."""
    parts = []
    total = 0
    for i, b in enumerate(blocks, 1):
        head = f"### Source {i}: {b.get('doc_path','')}\n"
        txt = str(b.get('text','')).strip()
        chunk = (head + txt).strip()
        if total + len(chunk) > max_chars and parts:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts) or "(no context)"

def _citations_footer(blocks: List[Dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, 1):
        lines.append(f"{_format_source(b, i)}")
    return "\n".join(lines)

def answer_with_citations(question: str, context_blocks: List[Dict]) -> Dict[str, str]:
    """RAG answer using OpenAI if available; always returns answer text."""
    sources = context_blocks or []
    ctx_text = build_context(sources)
    foot = _citations_footer(sources)
    system = (
        "You are a concise assistant. Answer from the provided CONTEXT only. "
        "If the answer is not present, say you don't know. "
        "Quote short phrases rather than long paragraphs. "
        "End your answer with a 'Sources:' section listing [n] doc and chunk id for each source you used."
    )
    user_prompt = (
        f"CONTEXT:\n{ctx_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Write a helpful, precise answer in the same language as the question. "
        f"Keep it under 10 sentences.\n"
        f"Then add:\nSources:\n{foot}"
    )

    if OPENAI_AVAILABLE:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        txt = resp.choices[0].message.content.strip()
        return {"answer": txt}

    # Fallback when no API key: stitch a simple non-LLM answer
    preview = sources[0]["text"][:400] + ("..." if len(sources[0]["text"]) > 400 else "") if sources else ""
    fallback = (
        "⚠️ OpenAI API key is not configured, so here is a heuristic answer template.\n\n"
        f"Question: {question}\n\n"
        "Likely relevant fragments from context:\n"
        f"> {preview}\n\n"
        "Sources:\n" + foot
    )
    return {"answer": fallback}
