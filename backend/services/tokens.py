"""Tokenización con tiktoken — para visualizar bajo el capó."""
from typing import List, Dict, Any
import tiktoken

_encodings: Dict[str, Any] = {}


def _encoding_for(model: str):
    if model not in _encodings:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        _encodings[model] = enc
    return _encodings[model]


def tokenize(text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    enc = _encoding_for(model)
    ids: List[int] = enc.encode(text)
    pieces: List[str] = []
    for tid in ids:
        try:
            pieces.append(enc.decode([tid]))
        except Exception:
            pieces.append("�")
    return {
        "model": model,
        "encoding": enc.name,
        "count": len(ids),
        "tokens": [{"id": tid, "text": piece} for tid, piece in zip(ids, pieces)],
    }
