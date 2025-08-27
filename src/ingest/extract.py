import os
from typing import List, Dict
from bs4 import BeautifulSoup
import html2text
from pypdf import PdfReader

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    return h.handle(str(soup))

def read_md_or_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

EXT_READERS = {
    ".pdf": read_pdf,
    ".html": read_html,
    ".htm": read_html,
    ".md": read_md_or_txt,
    ".txt": read_md_or_txt,
}

def load_documents(root: str) -> List[Dict]:
    docs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn.lower())[1]
            if ext in EXT_READERS:
                path = os.path.join(dirpath, fn)
                try:
                    text = EXT_READERS[ext](path)
                    if text and text.strip():
                        docs.append({"path": path, "text": text})
                except Exception as e:
                    print(f"[WARN] Failed to read {path}: {e}")
    return docs
