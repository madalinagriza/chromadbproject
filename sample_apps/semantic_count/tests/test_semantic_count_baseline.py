from __future__ import annotations

import numpy as np

from sample_apps.semantic_count.semantic_count_baseline import (
    infer_metric_from_dataset_path,
    semantic_count_collection,
    semantic_count_embeddings,
)


class FakeCollection:
    def __init__(self, ids: list[str], embeddings: list[list[float]]) -> None:
        self._ids = ids
        self._embeddings = embeddings

    def count(self) -> int:
        return len(self._ids)

    def get(self, *, limit: int, offset: int, include: list[str]) -> dict[str, object]:
        del include
        stop = offset + limit
        return {
            "ids": self._ids[offset:stop],
            "embeddings": self._embeddings[offset:stop],
        }


def test_semantic_count_embeddings_cosine_similarity() -> None:
    query = [1.0, 0.0]
    candidates = [
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    result = semantic_count_embeddings(
        query,
        candidates,
        threshold=0.7,
        metric="angular",
        threshold_mode="similarity",
    )

    assert result.count == 2
    assert result.total_candidates == 4
    assert result.metric == "cosine"


def test_semantic_count_embeddings_l2_distance() -> None:
    query = [0.0, 0.0]
    candidates = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
    ]

    result = semantic_count_embeddings(
        query,
        candidates,
        threshold=1.0,
        metric="euclidean",
        threshold_mode="distance",
    )

    assert result.count == 2
    assert result.metric == "l2"


def test_semantic_count_collection_batches_and_exclude_id() -> None:
    collection = FakeCollection(
        ids=["a", "b", "c", "d"],
        embeddings=[
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
    )

    result = semantic_count_collection(
        collection,
        np.array([1.0, 0.0], dtype=np.float32),
        threshold=0.9,
        metric="cosine",
        threshold_mode="similarity",
        batch_size=2,
        exclude_id="a",
    )

    assert result.count == 1
    assert result.total_candidates == 3


def test_infer_metric_from_dataset_path() -> None:
    assert infer_metric_from_dataset_path("glove-100-angular.hdf5") == "cosine"
    assert infer_metric_from_dataset_path("sift-128-euclidean.hdf5") == "l2"
