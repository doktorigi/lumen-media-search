from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .domain import MediaKind, SearchItem
from .embeddings import EmbeddingEngine
from .ranker import ExactRanker
from .store import MediaStore


@dataclass(frozen=True, slots=True)
class SearchResult:
    asset_id: int
    segment_id: int
    kind: str
    name: str
    path: str
    score: float
    timestamp_ms: int
    width: int | None
    height: int | None
    duration_ms: int | None
    modified_at: str
    thumbnail_url: str | None
    media_url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    elapsed_ms: int
    ranker: str
    results: tuple[SearchResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "elapsed_ms": self.elapsed_ms,
            "ranker": self.ranker,
            "results": [result.to_dict() for result in self.results],
        }


class SemanticSearch:
    def __init__(
        self,
        store: MediaStore,
        engine: EmbeddingEngine,
        ranker: ExactRanker | None = None,
    ):
        self.store = store
        self.engine = engine
        self.ranker = ranker or ExactRanker()

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        limit: int = 60,
    ) -> SearchResponse:
        started = time.perf_counter()
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search text cannot be empty")
        if kind in (None, "", "all"):
            media_kind = None
        elif kind in (MediaKind.IMAGE.value, MediaKind.VIDEO.value):
            media_kind = MediaKind(kind)
        else:
            raise ValueError("kind must be all, image, or video")

        bounded_limit = min(200, max(1, int(limit)))
        items, matrix = self.store.load_search_matrix(kind=media_kind)
        if not items:
            return SearchResponse(
                query=normalized_query,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                ranker=self.ranker.backend,
                results=tuple(),
            )

        query_embedding = self.engine.encode_text(normalized_query)
        # Pull enough segment candidates to avoid a long video monopolizing the
        # result grid, then collapse them to one card per asset.
        candidate_count = min(len(items), max(500, bounded_limit * 12))
        ranked = self.ranker.top_k(matrix, query_embedding, candidate_count)

        grouped: dict[int, list[tuple[float, SearchItem]]] = {}
        for row_index, score in zip(
            ranked.indices.tolist(), ranked.scores.tolist(), strict=True
        ):
            item = items[row_index]
            grouped.setdefault(item.asset_id, []).append((float(score), item))

        collapsed: list[tuple[float, SearchItem]] = []
        for matches in grouped.values():
            matches.sort(key=lambda match: (-match[0], match[1].timestamp_ms))
            best_score, best_item = matches[0]
            if best_item.kind == MediaKind.VIDEO:
                top_scores = [match[0] for match in matches[:3]]
                aggregate = 0.85 * best_score + 0.15 * (
                    sum(top_scores) / len(top_scores)
                )
            else:
                aggregate = best_score
            collapsed.append((aggregate, best_item))

        collapsed.sort(key=lambda value: (-value[0], value[1].asset_id))
        results = tuple(
            self._make_result(item, score) for score, item in collapsed[:bounded_limit]
        )
        return SearchResponse(
            query=normalized_query,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            ranker=self.ranker.backend,
            results=results,
        )

    @staticmethod
    def _make_result(item: SearchItem, score: float) -> SearchResult:
        modified = datetime.fromtimestamp(item.mtime_ns / 1_000_000_000).astimezone()
        return SearchResult(
            asset_id=item.asset_id,
            segment_id=item.segment_id,
            kind=item.kind.value,
            name=Path(item.path).name,
            path=item.path,
            score=round(float(score), 6),
            timestamp_ms=item.timestamp_ms,
            width=item.width,
            height=item.height,
            duration_ms=item.duration_ms,
            modified_at=modified.isoformat(),
            thumbnail_url=(
                f"/api/thumbnail/{item.segment_id}"
                if item.thumbnail_path is not None
                else None
            ),
            media_url=f"/api/media/{item.asset_id}",
        )
