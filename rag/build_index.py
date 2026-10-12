"""
Build the RAG retrieval index over the policy docs + generated retro notes.

Run from the project root: `python rag/build_index.py`

Embeds every document with a local sentence-transformers model (no API
needed for embedding — only the generation step at query time calls Groq),
and saves the embedding matrix + metadata for cosine-similarity retrieval.

Saves:
  rag/index/embeddings.npy
  rag/index/documents.json   [{id, source, type, text}, ...] aligned to embeddings.npy rows
"""

import glob
import json

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_docs():
    docs = []
    for path in sorted(glob.glob("docs/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": path.replace("\\", "/"), "type": "policy", "text": f.read().strip()})
    for path in sorted(glob.glob("docs/retros/*.md")):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": path.replace("\\", "/"), "type": "retro", "text": f.read().strip()})
    for i, d in enumerate(docs):
        d["id"] = i
    return docs


def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} documents ({sum(d['type']=='policy' for d in docs)} policy, "
          f"{sum(d['type']=='retro' for d in docs)} retro)")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    np.save("rag/index/embeddings.npy", embeddings.astype("float32"))
    with open("rag/index/documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    print(f"Saved embeddings {embeddings.shape} and documents.json")


if __name__ == "__main__":
    main()
