"""
Build the RAG retrieval index over the policy docs + generated retro notes.

Run from the project root: `python rag/build_index.py`

Embeds every document with a local ONNX embedding model via `fastembed`
(BAAI/bge-small-en-v1.5) — deliberately NOT `sentence-transformers`, which
pulls in PyTorch. Loading PyTorch alongside TensorFlow (needed for the
attendance/sprint models) in the same process segfaults on memory-
constrained deployment containers; fastembed runs on plain ONNX Runtime
(CPU), has no PyTorch dependency, and gives the same quality of semantic
retrieval — verified against a TF-IDF fallback, which missed the relevant
retro note for a "why did Team X underperform" query because it couldn't
match "underperform" against "underperformed" (pure keyword matching has
no sense of word morphology); dense embeddings handle that naturally.

Saves:
  rag/index/embeddings.npy
  rag/index/documents.json   [{id, source, type, text}, ...] aligned to embeddings.npy rows
"""

import glob
import json

import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


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

    model = TextEmbedding(model_name=EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = np.array(list(model.embed(texts)))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save("rag/index/embeddings.npy", embeddings.astype("float32"))
    with open("rag/index/documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    print(f"Saved embeddings {embeddings.shape} and documents.json")


if __name__ == "__main__":
    main()
