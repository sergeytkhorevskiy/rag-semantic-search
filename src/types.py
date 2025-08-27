from typing import TypedDict

class Hit(TypedDict):
    score: float
    text: str
    chunk_id: str
    doc_path: str
