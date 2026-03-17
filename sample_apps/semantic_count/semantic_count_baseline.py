from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Sequence

import numpy as np

ThresholdMode = Literal["similarity", "distance"]

_METRIC_ALIASES = {
    "angular": "cosine",
    "cosine": "cosine",
    "euclidean": "l2",
    "l2": "l2",
    "inner_product": "ip",
    "ip": "ip",
}


@dataclass(frozen=True)
class SemanticCountResult:
    count: int
    elapsed_ms: float
    total_candidates: int
    metric: str
    threshold: float
    threshold_mode: ThresholdMode


def _canonical_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized not in _METRIC_ALIASES:
        supported = ", ".join(sorted(_METRIC_ALIASES))
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of: {supported}")
    return _METRIC_ALIASES[normalized]


def infer_metric_from_dataset_path(dataset_path: str | Path) -> str:
    name = Path(dataset_path).name.lower()
    if "angular" in name or "cosine" in name:
        return "cosine"
    if "euclidean" in name or "l2" in name:
        return "l2"
    if "inner-product" in name or "inner_product" in name or "-ip" in name:
        return "ip"
    raise ValueError(
        "Could not infer the metric from the dataset filename. Pass --metric explicitly."
    )


def _ensure_1d(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D query embedding, got shape {array.shape}")
    return array


def _ensure_2d(embeddings: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
    return array


def _cosine_similarity(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    eps = np.float32(1e-30)
    denominators = np.maximum(query_norm * candidate_norms, eps)
    return np.matmul(candidate_embeddings, query_embedding) / denominators


def _score_candidates(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    metric: str,
    threshold_mode: ThresholdMode,
) -> np.ndarray:
    canonical_metric = _canonical_metric(metric)

    if threshold_mode == "similarity":
        if canonical_metric == "cosine":
            return _cosine_similarity(query_embedding, candidate_embeddings)
        if canonical_metric == "ip":
            return np.matmul(candidate_embeddings, query_embedding)
        raise ValueError(
            "Similarity thresholds are only supported for cosine/angular and ip metrics. "
            "Use threshold_mode='distance' for l2/euclidean datasets."
        )

    if canonical_metric == "cosine":
        return 1.0 - _cosine_similarity(query_embedding, candidate_embeddings)
    if canonical_metric == "ip":
        return 1.0 - np.matmul(candidate_embeddings, query_embedding)
    if canonical_metric == "l2":
        deltas = candidate_embeddings - query_embedding
        return np.sum(deltas * deltas, axis=1)

    raise ValueError(f"Unsupported metric '{metric}'")


def _count_scores(scores: np.ndarray, threshold: float, threshold_mode: ThresholdMode) -> int:
    if threshold_mode == "similarity":
        return int(np.count_nonzero(scores >= threshold))
    return int(np.count_nonzero(scores <= threshold))


def semantic_count_embeddings(
    query_embedding: Sequence[float] | np.ndarray,
    candidate_embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    threshold: float,
    metric: str = "cosine",
    threshold_mode: ThresholdMode = "similarity",
) -> SemanticCountResult:
    query_array = _ensure_1d(query_embedding)
    candidate_array = _ensure_2d(candidate_embeddings)
    if candidate_array.shape[1] != query_array.shape[0]:
        raise ValueError(
            "Embedding dimensionality mismatch: "
            f"query has dim {query_array.shape[0]} but candidates have dim {candidate_array.shape[1]}"
        )

    start = time.perf_counter()
    scores = _score_candidates(query_array, candidate_array, metric, threshold_mode)
    count = _count_scores(scores, threshold, threshold_mode)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return SemanticCountResult(
        count=count,
        elapsed_ms=elapsed_ms,
        total_candidates=int(candidate_array.shape[0]),
        metric=_canonical_metric(metric),
        threshold=threshold,
        threshold_mode=threshold_mode,
    )


def load_query_embedding_from_collection(collection: Any, query_id: str) -> np.ndarray:
    result = collection.get(ids=[query_id], include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise ValueError(f"Could not load embedding for query id '{query_id}'")
    return _ensure_1d(embeddings[0])


def semantic_count_collection(
    collection: Any,
    query_embedding: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    metric: str = "cosine",
    threshold_mode: ThresholdMode = "similarity",
    batch_size: int = 2048,
    exclude_id: Optional[str] = None,
) -> SemanticCountResult:
    query_array = _ensure_1d(query_embedding)
    total_candidates = int(collection.count())
    total_count = 0
    excluded_present = 0
    start = time.perf_counter()

    for offset in range(0, total_candidates, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings"],
        )
        embeddings = batch.get("embeddings")
        ids = batch.get("ids") or []
        if embeddings is None:
            raise ValueError("Collection batch did not include embeddings")
        if len(embeddings) == 0:
            continue

        batch_array = _ensure_2d(embeddings)
        scores = _score_candidates(query_array, batch_array, metric, threshold_mode)
        batch_count = _count_scores(scores, threshold, threshold_mode)

        if exclude_id is not None:
            for idx, record_id in enumerate(ids):
                if record_id == exclude_id:
                    excluded_present = 1
                    if threshold_mode == "similarity" and scores[idx] >= threshold:
                        batch_count -= 1
                    elif threshold_mode == "distance" and scores[idx] <= threshold:
                        batch_count -= 1

        total_count += batch_count

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SemanticCountResult(
        count=total_count,
        elapsed_ms=elapsed_ms,
        total_candidates=total_candidates - excluded_present,
        metric=_canonical_metric(metric),
        threshold=threshold,
        threshold_mode=threshold_mode,
    )


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required for ANN Benchmarks .hdf5 files. "
            "Install it with: .\\venv\\Scripts\\python.exe -m pip install -r sample_apps\\semantic_count\\requirements.txt"
        ) from exc
    return h5py


def inspect_ann_benchmark_dataset(dataset_path: str | Path) -> Dict[str, Dict[str, Any]]:
    h5py = _require_h5py()
    path = Path(dataset_path)
    summary: Dict[str, Dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            value = handle[key]
            summary[key] = {
                "shape": tuple(int(part) for part in value.shape),
                "dtype": str(value.dtype),
            }
    return summary


def _iter_hdf5_batches(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    total_rows = int(dataset.shape[0])
    for start in range(0, total_rows, batch_size):
        stop = min(start + batch_size, total_rows)
        yield start, np.asarray(dataset[start:stop], dtype=np.float32)


def semantic_count_ann_benchmark(
    dataset_path: str | Path,
    *,
    query_index: int,
    threshold: float,
    query_split: str = "test",
    target_split: str = "train",
    metric: Optional[str] = None,
    threshold_mode: ThresholdMode = "similarity",
    batch_size: int = 8192,
    exclude_self: bool = False,
) -> SemanticCountResult:
    h5py = _require_h5py()
    path = Path(dataset_path)
    metric_name = metric or infer_metric_from_dataset_path(path)

    with h5py.File(path, "r") as handle:
        query_dataset = handle[query_split]
        target_dataset = handle[target_split]
        query_embedding = _ensure_1d(np.asarray(query_dataset[query_index], dtype=np.float32))
        total_candidates = int(target_dataset.shape[0])
        total_count = 0
        start = time.perf_counter()

        for batch_start, batch_embeddings in _iter_hdf5_batches(target_dataset, batch_size):
            scores = _score_candidates(
                query_embedding,
                batch_embeddings,
                metric_name,
                threshold_mode,
            )
            batch_count = _count_scores(scores, threshold, threshold_mode)

            if exclude_self and query_split == target_split:
                relative_index = query_index - batch_start
                if 0 <= relative_index < len(scores):
                    score = scores[relative_index]
                    if threshold_mode == "similarity" and score >= threshold:
                        batch_count -= 1
                    elif threshold_mode == "distance" and score <= threshold:
                        batch_count -= 1

            total_count += batch_count

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SemanticCountResult(
        count=total_count,
        elapsed_ms=elapsed_ms,
        total_candidates=total_candidates - (1 if exclude_self else 0),
        metric=_canonical_metric(metric_name),
        threshold=threshold,
        threshold_mode=threshold_mode,
    )


def ingest_ann_benchmark_split(
    dataset_path: str | Path,
    *,
    persist_dir: str | Path,
    collection_name: str,
    split: str = "train",
    metric: Optional[str] = None,
    batch_size: int = 2048,
    reset_collection: bool = False,
) -> Dict[str, Any]:
    import chromadb

    h5py = _require_h5py()
    path = Path(dataset_path)
    persist_path = Path(persist_dir)
    metric_name = _canonical_metric(metric or infer_metric_from_dataset_path(path))

    client = chromadb.PersistentClient(path=str(persist_path))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": metric_name},
    )

    inserted = 0
    with h5py.File(path, "r") as handle:
        dataset = handle[split]
        total_rows = int(dataset.shape[0])
        for batch_start, batch_embeddings in _iter_hdf5_batches(dataset, batch_size):
            ids = [
                f"{split}-{row_index:08d}"
                for row_index in range(batch_start, batch_start + len(batch_embeddings))
            ]
            metadatas = [
                {
                    "dataset": path.name,
                    "split": split,
                    "row_index": row_index,
                }
                for row_index in range(batch_start, batch_start + len(batch_embeddings))
            ]
            collection.add(
                ids=ids,
                embeddings=batch_embeddings.tolist(),
                metadatas=metadatas,
            )
            inserted += len(batch_embeddings)

    return {
        "collection": collection_name,
        "metric": metric_name,
        "persist_dir": str(persist_path),
        "rows_inserted": inserted,
        "split": split,
        "total_rows": total_rows,
    }


def _load_query_vector_from_file(path: str | Path) -> np.ndarray:
    query_path = Path(path)
    if query_path.suffix == ".npy":
        return _ensure_1d(np.load(query_path))
    if query_path.suffix == ".json":
        return _ensure_1d(json.loads(query_path.read_text()))
    raw_text = query_path.read_text().strip()
    if raw_text.startswith("["):
        return _ensure_1d(json.loads(raw_text))
    values = [float(part) for part in raw_text.split(",") if part.strip()]
    return _ensure_1d(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact semantic_count baseline and ANN Benchmarks helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-hdf5",
        help="Inspect the layout of an ANN Benchmarks HDF5 dataset.",
    )
    inspect_parser.add_argument("--dataset", required=True, help="Path to the HDF5 dataset.")

    count_hdf5_parser = subparsers.add_parser(
        "count-hdf5",
        help="Run exact semantic_count over an ANN Benchmarks HDF5 dataset.",
    )
    count_hdf5_parser.add_argument("--dataset", required=True, help="Path to the HDF5 dataset.")
    count_hdf5_parser.add_argument("--query-index", required=True, type=int)
    count_hdf5_parser.add_argument("--query-split", default="test")
    count_hdf5_parser.add_argument("--target-split", default="train")
    count_hdf5_parser.add_argument("--threshold", required=True, type=float)
    count_hdf5_parser.add_argument(
        "--threshold-mode",
        choices=["similarity", "distance"],
        default="similarity",
    )
    count_hdf5_parser.add_argument("--metric")
    count_hdf5_parser.add_argument("--batch-size", type=int, default=8192)
    count_hdf5_parser.add_argument("--exclude-self", action="store_true")

    ingest_parser = subparsers.add_parser(
        "ingest-hdf5",
        help="Ingest one ANN Benchmarks split into a persistent Chroma collection.",
    )
    ingest_parser.add_argument("--dataset", required=True, help="Path to the HDF5 dataset.")
    ingest_parser.add_argument("--persist-dir", required=True)
    ingest_parser.add_argument("--collection", required=True)
    ingest_parser.add_argument("--split", default="train")
    ingest_parser.add_argument("--metric")
    ingest_parser.add_argument("--batch-size", type=int, default=2048)
    ingest_parser.add_argument("--reset", action="store_true")

    count_collection_parser = subparsers.add_parser(
        "count-collection",
        help="Run exact semantic_count over a Chroma collection by paging through embeddings.",
    )
    count_collection_parser.add_argument("--persist-dir", required=True)
    count_collection_parser.add_argument("--collection", required=True)
    count_collection_parser.add_argument("--threshold", required=True, type=float)
    count_collection_parser.add_argument(
        "--threshold-mode",
        choices=["similarity", "distance"],
        default="similarity",
    )
    count_collection_parser.add_argument("--metric", default="cosine")
    count_collection_parser.add_argument("--query-id")
    count_collection_parser.add_argument("--query-vector-file")
    count_collection_parser.add_argument("--batch-size", type=int, default=2048)
    count_collection_parser.add_argument("--exclude-query-id", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "inspect-hdf5":
        print(json.dumps(inspect_ann_benchmark_dataset(args.dataset), indent=2))
        return

    if args.command == "count-hdf5":
        result = semantic_count_ann_benchmark(
            args.dataset,
            query_index=args.query_index,
            query_split=args.query_split,
            target_split=args.target_split,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            metric=args.metric,
            batch_size=args.batch_size,
            exclude_self=args.exclude_self,
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "ingest-hdf5":
        result = ingest_ann_benchmark_split(
            args.dataset,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            split=args.split,
            metric=args.metric,
            batch_size=args.batch_size,
            reset_collection=args.reset,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "count-collection":
        import chromadb

        if not args.query_id and not args.query_vector_file:
            parser.error("count-collection requires either --query-id or --query-vector-file")

        client = chromadb.PersistentClient(path=args.persist_dir)
        collection = client.get_collection(args.collection)
        exclude_id = None

        if args.query_id:
            query_embedding = load_query_embedding_from_collection(collection, args.query_id)
            if args.exclude_query_id:
                exclude_id = args.query_id
        else:
            query_embedding = _load_query_vector_from_file(args.query_vector_file)

        result = semantic_count_collection(
            collection,
            query_embedding,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            metric=args.metric,
            batch_size=args.batch_size,
            exclude_id=exclude_id,
        )
        print(json.dumps(asdict(result), indent=2))
        return

    parser.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
