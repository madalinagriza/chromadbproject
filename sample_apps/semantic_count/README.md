# Semantic Count Baseline

This folder contains the first milestone from the project plan: an exact
`semantic_count(query, threshold)` baseline that we can use as ground truth
before implementing the sampling, iterative probing, and LSH estimators.

## Where The Baseline Should Run

After inspecting both the Chroma codebase and the ANN Benchmarks dataset format,
the cleanest place to run the brute-force baseline is:

1. **First on the raw ANN dataset file** (`glove-100-angular.hdf5`,
   `sift-128-euclidean.hdf5`, etc.). This keeps the ground truth independent
   from HNSW or any other ANN index behavior.
2. **Then on a Chroma collection through paginated
   `collection.get(..., include=["embeddings"])`.** This lets us benchmark the
   same exact-count logic after ingestion, without modifying the underlying
   index.

That split matches the proposal: Chroma is the system under evaluation, while
the brute-force count remains an exact scan.

## Files

- `semantic_count_baseline.py` Exact-count helpers plus a CLI with dataset
  inspection, ingestion, and exact count commands.
- `requirements.txt` Extra dependency needed for ANN Benchmarks `.hdf5` files.

## Setup

From the repo root:

```powershell
.\venv\Scripts\python.exe -m pip install -r sample_apps\semantic_count\requirements.txt
```

## Inspect The ANN Dataset

```powershell
.\venv\Scripts\python.exe -m sample_apps.semantic_count.semantic_count_baseline inspect-hdf5 `
  --dataset glove-100-angular.hdf5
```

This should show the standard ANN Benchmarks layout:

- `train`: candidate vectors to index in Chroma
- `test`: held-out query vectors
- `neighbors` and `distances`: precomputed nearest-neighbor reference data

## Run The Exact Baseline Directly On The Dataset

```powershell
.\venv\Scripts\python.exe -m sample_apps.semantic_count.semantic_count_baseline count-hdf5 --dataset glove-100-angular.hdf5 --query-index 0 --threshold 0.4 --threshold-mode similarity --metric angular
```

For `angular` datasets, the baseline treats the score as cosine similarity,
which matches how Chroma maps this space in practice.

Example output from the verified local run:

```text
(venv) C:\Users\mgriz\Documents\chromadbproject>.\venv\Scripts\python.exe -m sample_apps.semantic_count.semantic_count_baseline count-hdf5 --dataset glove-100-angular.hdf5 --query-index 0 --threshold 0.4 --threshold-mode similarity --metric angular
<frozen runpy>:128: RuntimeWarning: 'sample_apps.semantic_count.semantic_count_baseline' found in sys.modules after import of package 'sample_apps.semantic_count', but prior to execution of 'sample_apps.semantic_count.semantic_count_baseline'; this may result in unpredictable behaviour
{
  "count": 7407,
  "elapsed_ms": 456.138200010173,
  "total_candidates": 1183514,
  "metric": "cosine",
  "threshold": 0.4,
  "threshold_mode": "similarity"
}
```

## Optional: Ingest The Train Split Into Chroma

```powershell
.\venv\Scripts\python.exe -m sample_apps.semantic_count.semantic_count_baseline ingest-hdf5 `
  --dataset glove-100-angular.hdf5 `
  --persist-dir sample_apps\semantic_count\chroma_data `
  --collection glove_train `
  --split train `
  --metric angular `
  --reset
```

This stores the vectors in a persistent Chroma collection using the matching
HNSW space metadata.

## Run The Exact Baseline Over A Chroma Collection

If the query vector is already in the collection:

```powershell
.\venv\Scripts\python.exe -m sample_apps.semantic_count.semantic_count_baseline count-collection `
  --persist-dir sample_apps\semantic_count\chroma_data `
  --collection glove_train `
  --query-id train-00000000 `
  --threshold 0.6 `
  --threshold-mode similarity `
  --metric cosine `
  --exclude-query-id
```

If you want to query the collection with a held-out vector, pass a `.json`,
`.npy`, or `.txt` vector file with `--query-vector-file`.

## Suggested Next Step

Once the rest of the data is in place, use this exact baseline to generate a
fixed workload of 100-500 random test queries and save the counts plus
latencies. The estimators can then be compared against the same workload using
Q-error and speedup.
