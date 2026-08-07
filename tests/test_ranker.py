from __future__ import annotations

import numpy as np
import pytest

from lumen_media.ranker import ExactRanker


@pytest.fixture
def ranker() -> ExactRanker:
    return ExactRanker(prefer_mojo=False)


def test_numpy_ranker_returns_descending_scores_and_source_indices(
    ranker: ExactRanker,
) -> None:
    matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = ranker.top_k(matrix, np.array([1.0, 0.0]), k=3)

    assert ranker.backend == "NumPy BLAS"
    np.testing.assert_array_equal(result.indices, np.array([1, 2, 0]))
    np.testing.assert_allclose(result.scores, np.array([1.0, 0.5, 0.0]))
    assert result.indices.dtype == np.int64
    assert result.scores.dtype == np.float32


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.ones((1, 2, 3), dtype=np.float32),
    ],
)
def test_numpy_ranker_rejects_non_matrix_inputs(
    ranker: ExactRanker, matrix: np.ndarray
) -> None:
    with pytest.raises(
        ValueError, match=r"matrix must have shape \[rows, dimensions\]"
    ):
        ranker.top_k(matrix, np.array([1.0, 0.0]), k=1)


def test_numpy_ranker_rejects_query_dimension_mismatch(
    ranker: ExactRanker,
) -> None:
    with pytest.raises(ValueError, match="query dimensions do not match"):
        ranker.top_k(np.ones((2, 3)), np.ones(2), k=1)


def test_numpy_ranker_handles_empty_search_matrix(ranker: ExactRanker) -> None:
    result = ranker.top_k(
        np.empty((0, 3), dtype=np.float32),
        np.ones(3, dtype=np.float32),
        k=10,
    )

    assert result.indices.shape == (0,)
    assert result.scores.shape == (0,)
    assert result.indices.dtype == np.int64
    assert result.scores.dtype == np.float32


@pytest.mark.parametrize("k", [0, -1])
def test_numpy_ranker_treats_non_positive_k_as_empty(
    ranker: ExactRanker, k: int
) -> None:
    result = ranker.top_k(np.eye(2), np.array([1.0, 0.0]), k=k)

    assert result.indices.size == 0
    assert result.scores.size == 0


def test_numpy_ranker_caps_k_at_row_count(ranker: ExactRanker) -> None:
    matrix = np.array([[0.25], [1.0], [-0.5]], dtype=np.float32)

    result = ranker.top_k(matrix, np.array([1.0]), k=99)

    np.testing.assert_array_equal(result.indices, np.array([1, 0, 2]))
    np.testing.assert_allclose(result.scores, np.array([1.0, 0.25, -0.5]))


def test_numpy_ranker_breaks_cutoff_ties_by_source_order(
    ranker: ExactRanker,
) -> None:
    matrix = np.ones((5, 1), dtype=np.float32)

    result = ranker.top_k(matrix, np.array([1.0]), k=3)

    np.testing.assert_array_equal(result.indices, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.scores, np.ones(3, dtype=np.float32))
