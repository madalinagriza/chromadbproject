"""
Download a subset of MS MARCO passages and embed them with SentenceTransformers.
Saves embeddings to msmarco_embeddings.npy and passages to msmarco_passages.json.

NOTE: Pre-computed embeddings and passages are available on Google Drive — no need to run this script.
Download msmarco_embeddings.npy and msmarco_passages.json from:
https://drive.google.com/drive/folders/1Cdh7B33UtmO5xh-rceekhYPBGCZHJBw6?usp=sharing
Place both files in the repo root before running experiments.
"""

import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

N = 100_000          # number of passages to embed
BATCH_SIZE = 512
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast
OUT_EMBEDDINGS = "msmarco_embeddings.npy"
OUT_PASSAGES = "msmarco_passages.json"

print(f"Loading MS MARCO passage corpus (first {N:,} passages)...")
ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True)

passages = []
for i, example in enumerate(tqdm(ds, total=N, desc="Reading passages")):
    if i >= N:
        break
    # Each example has a 'passages' dict with 'passage_text' list
    for text in example["passages"]["passage_text"]:
        passages.append(text)
        if len(passages) >= N:
            break
    if len(passages) >= N:
        break

print(f"Collected {len(passages):,} passages. Embedding with {MODEL_NAME}...")

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    passages,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # cosine similarity via dot product
)

np.save(OUT_EMBEDDINGS, embeddings)
with open(OUT_PASSAGES, "w") as f:
    json.dump(passages, f)

print(f"Saved {embeddings.shape} embeddings to {OUT_EMBEDDINGS}")
print(f"Saved passages to {OUT_PASSAGES}")
