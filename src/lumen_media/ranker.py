from __future__ import annotations

import importlib
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RankResult:
    indices: np.ndarray
    scores: np.ndarray


class ExactRanker:
    """Exact cosine top-k with a Mojo SIMD kernel and a NumPy fallback."""

    def __init__(self, *, prefer_mojo: bool = True):
        self._mojo_module = self._load_mojo() if prefer_mojo else None
        self.backend = "Mojo SIMD" if self._mojo_module is not None else "NumPy BLAS"

    @staticmethod
    def _load_mojo():
        try:
            import mojo.importer  # type: ignore[import-not-found]  # noqa: F401
        except (ImportError, OSError):
            return None

        kernel_dir = Path(__file__).with_name("mojo_kernels")
        kernel_path = str(kernel_dir)
        inserted = kernel_path not in sys.path
        if inserted:
            sys.path.insert(0, kernel_path)
        try:
            return importlib.import_module("mojo_ranker")
        except Exception:
            return None
        finally:
            if inserted:
                with suppress(ValueError):
                    sys.path.remove(kernel_path)

    def top_k(self, matrix: np.ndarray, query: np.ndarray, k: int) -> RankResult:
        values = np.ascontiguousarray(matrix, dtype=np.float32)
        needle = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        if values.ndim != 2:
            raise ValueError("matrix must have shape [rows, dimensions]")
        if values.shape[1] != needle.shape[0]:
            raise ValueError("query dimensions do not match the search matrix")
        count = min(max(0, int(k)), values.shape[0])
        if count == 0:
            return RankResult(
                indices=np.empty(0, dtype=np.int64),
                scores=np.empty(0, dtype=np.float32),
            )

        if self._mojo_module is not None:
            try:
                indices, scores = self._mojo_module.top_k(values, needle, count)
                return RankResult(
                    indices=np.asarray(indices, dtype=np.int64),
                    scores=np.asarray(scores, dtype=np.float32),
                )
            except Exception:
                # The interop API is beta. A query should never fail just because the
                # optional compiled kernel is incompatible with a newer Mojo build.
                self._mojo_module = None
                self.backend = "NumPy BLAS"

        scores = np.asarray(values @ needle, dtype=np.float32)
        # The secondary source-index key makes ties deterministic, including
        # ties exactly at the k cutoff (argpartition cannot guarantee that).
        source_indices = np.arange(scores.size)
        indices = np.asarray(
            np.lexsort((source_indices, -scores))[:count], dtype=np.int64
        )
        return RankResult(indices=indices, scores=scores[indices])
