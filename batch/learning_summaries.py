from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, Field

from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


# ---------- SQL schema (authoritative) ----------
LEARNING_SUMMARIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_summaries (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    window_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    scope_key TEXT,
    metrics JSON NOT NULL,
    summary_text TEXT NOT NULL,
    computed_at TIMESTAMP NOT NULL
);
"""


# ---------- Pydantic row models ----------
class LearningSummaryRow(BaseModel):
    id: str
    user_id: str
    window_type: str
    scope: str
    scope_key: Optional[str]
    metrics: Dict[str, object]
    summary_text: str
    computed_at: datetime


class UserTextRow(BaseModel):
    id: str
    user_id: str
    session_id: Optional[str]
    text: str
    created_at: datetime


class MistakePatternRow(BaseModel):
    id: str
    user_id: str
    canonical_description: str
    features: Dict[str, str]
    created_at: datetime
    last_seen_at: datetime


class LessonArtifactRow(BaseModel):
    id: str
    user_id: str
    patterns_covered: List[str]
    pedagogy_tags: List[str]
    created_at: datetime


class SessionRow(BaseModel):
    id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]


class MistakeOccurrenceRow(BaseModel):
    pattern_id: str
    user_text_id: str
    detected_at: datetime


# ---------- In-memory SQL placeholder (authoritative) ----------
class InMemorySQLStore:
    """Authoritative storage placeholder. TODO: replace with real SQL."""

    def __init__(self):
        self.user_texts: List[Dict] = []
        self.mistake_patterns: List[Dict] = []
        self.mistake_occurrences: List[Dict] = []
        self.lesson_artifacts: List[Dict] = []
        self.sessions: List[Dict] = []
        self.learning_summaries: List[Dict] = []

    def upsert_learning_summary(self, row: LearningSummaryRow) -> None:
        self.learning_summaries = [
            r
            for r in self.learning_summaries
            if not (
                r["user_id"] == row.user_id
                and r["window_type"] == row.window_type
                and r["scope"] == row.scope
                and r.get("scope_key") == row.scope_key
            )
        ]
        self.learning_summaries.append(row.model_dump())

    def load_df(self, table_name: str) -> pl.DataFrame:
        data = getattr(self, table_name, [])
        if not data:
            return pl.DataFrame()
        return pl.DataFrame(data)


# ---------- Helpers ----------
WINDOW_DEFS: Dict[str, int] = {
    "rolling_7d": 7,
    "rolling_30d": 30,
    "rolling_90d": 90,
}

SCOPE_GLOBAL = "global"
SCOPE_PATTERN = "pattern"


@dataclass
class BatchInputs:
    user_texts: pl.DataFrame
    mistake_patterns: pl.DataFrame
    mistake_occurrences: pl.DataFrame
    lesson_artifacts: pl.DataFrame
    sessions: pl.DataFrame


class LearningSummaryBatch:
    def __init__(self, sql_store: InMemorySQLStore, qdrant: QdrantStore, embedder: Embedder):
        self.sql_store = sql_store
        self.qdrant = qdrant
        self.embedder = embedder

    # ----- Public entrypoint -----
    def run(self, as_of: Optional[datetime] = None) -> None:
        as_of = as_of or datetime.utcnow()

        inputs = self._load_inputs()
        if inputs.user_texts.is_empty():
            return

        normalized = self._normalize_time(inputs)
        joined = self._join_events(normalized)
        if joined.is_empty():
            return

        for window_type, days in WINDOW_DEFS.items():
            window_start = (as_of - timedelta(days=days)).date()
            window_end = as_of.date()
            window_df = self._slice_window(joined, window_start, window_end)
            if window_df.is_empty():
                continue

            global_summaries = self._compute_scope_global(
                window_df, window_type, window_start, as_of
            )
            pattern_summaries = self._compute_scope_pattern(
                window_df, window_type, window_start, as_of
            )

            all_summaries = global_summaries + pattern_summaries
            for summary in all_summaries:
                self.sql_store.upsert_learning_summary(summary)
                self._upsert_qdrant(summary)

    # ----- Load / normalize -----
    def _load_inputs(self) -> BatchInputs:
        return BatchInputs(
            user_texts=self.sql_store.load_df("user_texts"),
            mistake_patterns=self.sql_store.load_df("mistake_patterns"),
            mistake_occurrences=self.sql_store.load_df("mistake_occurrences"),
            lesson_artifacts=self.sql_store.load_df("lesson_artifacts"),
            sessions=self.sql_store.load_df("sessions"),
        )

    def _normalize_time(self, inputs: BatchInputs) -> BatchInputs:
        def add_day(df: pl.DataFrame, ts_col: str) -> pl.DataFrame:
            if df.is_empty():
                return df
            return df.with_columns(
                pl.col(ts_col).dt.truncate("1d").alias("day"),
                pl.col(ts_col).dt.date().alias("date"),
            )

        return BatchInputs(
            user_texts=add_day(inputs.user_texts, "created_at"),
            mistake_patterns=add_day(inputs.mistake_patterns, "created_at"),
            mistake_occurrences=add_day(inputs.mistake_occurrences, "detected_at"),
            lesson_artifacts=add_day(inputs.lesson_artifacts, "created_at"),
            sessions=add_day(inputs.sessions, "started_at"),
        )

    # ----- Joins -----
    def _join_events(self, inputs: BatchInputs) -> pl.DataFrame:
        if inputs.user_texts.is_empty() or inputs.mistake_occurrences.is_empty():
            return pl.DataFrame()

        occ = inputs.mistake_occurrences
        texts = inputs.user_texts
        patterns = inputs.mistake_patterns

        base = (
            occ.join(texts, left_on="user_text_id", right_on="id", how="inner", suffix="_text")
            .join(patterns, left_on="pattern_id", right_on="id", how="left", suffix="_pattern")
        )

        return base.select(
            "user_id",
            "session_id",
            pl.col("pattern_id"),
            pl.col("canonical_description").alias("pattern_desc"),
            pl.col("features").alias("pattern_features"),
            pl.col("detected_at"),
            pl.col("day"),
            pl.col("date"),
        )

    # ----- Window slicing -----
    def _slice_window(self, df: pl.DataFrame, start_date, end_date) -> pl.DataFrame:
        return df.filter((pl.col("date") >= pl.lit(start_date)) & (pl.col("date") <= pl.lit(end_date)))

    # ----- Aggregations -----
    def _compute_scope_global(
        self, df: pl.DataFrame, window_type: str, window_start, as_of: datetime
    ) -> List[LearningSummaryRow]:
        grouped = (
            df.group_by(["user_id", "date"])
            .agg(pl.len().alias("occurrences"), pl.n_unique("session_id").alias("sessions"))
            .sort("date")
        )
        if grouped.is_empty():
            return []
        summaries = []
        for user_id, user_df in grouped.group_by("user_id"):
            metrics = self._compute_metrics(user_df, window_type, scope=SCOPE_GLOBAL)
            if metrics is None:
                metrics = {"cold_start": True}
                summary_text = self._cold_start_text()
            else:
                summary_text = self._build_summary_text(
                    window_type=window_type,
                    scope=SCOPE_GLOBAL,
                    pattern_desc="overall mistakes",
                    metrics=metrics,
                )
            summaries.append(
                LearningSummaryRow(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    window_type=window_type,
                    scope=SCOPE_GLOBAL,
                    scope_key=None,
                    metrics=metrics,
                    summary_text=summary_text,
                    computed_at=as_of,
                )
            )
        return summaries

    def _compute_scope_pattern(
        self, df: pl.DataFrame, window_type: str, window_start, as_of: datetime
    ) -> List[LearningSummaryRow]:
        filtered = df.drop_nulls("pattern_id")
        if filtered.is_empty():
            return []

        grouped = (
            filtered.group_by(["user_id", "pattern_id", "pattern_desc", "date"])
            .agg(pl.len().alias("occurrences"), pl.n_unique("session_id").alias("sessions"))
            .sort("date")
        )
        summaries: List[LearningSummaryRow] = []

        for keys, user_pattern_df in grouped.group_by(["user_id", "pattern_id", "pattern_desc"]):
            user_id = keys["user_id"]
            pattern_id = keys["pattern_id"]
            pattern_desc = keys["pattern_desc"] or "this pattern"
            metrics = self._compute_metrics(user_pattern_df, window_type, scope=SCOPE_PATTERN)
            if metrics is None:
                metrics = {"cold_start": True}
                summary_text = self._cold_start_text()
            else:
                summary_text = self._build_summary_text(
                    window_type=window_type,
                    scope=SCOPE_PATTERN,
                    pattern_desc=pattern_desc,
                    metrics=metrics,
                )
            summaries.append(
                LearningSummaryRow(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    window_type=window_type,
                    scope=SCOPE_PATTERN,
                    scope_key=pattern_id,
                    metrics=metrics,
                    summary_text=summary_text,
                    computed_at=as_of,
                )
            )

        return summaries

    # ----- Metrics -----
    def _compute_metrics(
        self, counts_df: pl.DataFrame, window_type: str, scope: str
    ) -> Optional[Dict[str, float]]:
        if counts_df.height < 2:
            return None

        counts_df = counts_df.with_columns(
            pl.col("occurrences").alias("count"),
            pl.arange(0, pl.count()).alias("t_idx"),
        )

        # frequency_trend: simple slope
        slope = self._slope(counts_df, "t_idx", "count")

        # error_rate_delta: last vs first normalized
        first_count = counts_df["count"][0]
        last_count = counts_df["count"][-1]
        denom = max(first_count, 1)
        error_rate_delta = (last_count - first_count) / denom

        # mastery_score: reward reduction and recency (lower slope, lower last_count)
        mastery_score = self._bounded(
            (self._neg(last_count) + self._neg(slope)) / 2.0
        )

        # stability_score: inverse variance
        variance = counts_df["count"].var()
        stability_score = self._bounded(1.0 / (1.0 + variance))

        # exposure_vs_improvement: sessions vs delta
        sessions = counts_df["sessions"] if "sessions" in counts_df.columns else pl.Series([])
        mean_sessions = sessions.mean() if not sessions.is_empty() else 0.0
        exposure_vs_improvement = self._bounded((mean_sessions - error_rate_delta) / 2.0)

        return {
            "frequency_trend": float(slope),
            "error_rate_delta": float(error_rate_delta),
            "mastery_score": float(mastery_score),
            "stability_score": float(stability_score),
            "exposure_vs_improvement": float(exposure_vs_improvement),
        }

    @staticmethod
    def _slope(df: pl.DataFrame, x_col: str, y_col: str) -> float:
        n = df.height
        x = df[x_col]
        y = df[y_col]
        sum_x = float(x.sum())
        sum_y = float(y.sum())
        sum_xy = float((x * y).sum())
        sum_x2 = float((x * x).sum())
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    @staticmethod
    def _bounded(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _neg(value: float) -> float:
        return -value

    # ----- Summary text -----
    def _build_summary_text(
        self,
        window_type: str,
        scope: str,
        pattern_desc: str,
        metrics: Dict[str, float],
    ) -> str:
        window_label = {
            "rolling_7d": "7 days",
            "rolling_30d": "30 days",
            "rolling_90d": "90 days",
        }.get(window_type, window_type)

        trend_desc = self._describe_trend(metrics["frequency_trend"])
        improvement_level = self._describe_improvement(metrics["error_rate_delta"])
        stability_desc = self._describe_stability(metrics["stability_score"])
        focus_hint = self._focus_hint(metrics)

        return (
            f"Over the last {window_label}, {pattern_desc} errors have {trend_desc}. "
            f"Improvement is {improvement_level}, with {stability_desc}. "
            f"Recommended focus: {focus_hint}."
        )

    @staticmethod
    def _describe_trend(freq_trend: float) -> str:
        if freq_trend < -0.3:
            return "decreased steadily"
        if freq_trend < -0.05:
            return "declined modestly"
        if freq_trend <= 0.05:
            return "remained stable"
        if freq_trend <= 0.3:
            return "increased slightly"
        return "increased sharply"

    @staticmethod
    def _describe_improvement(delta: float) -> str:
        if delta < -0.4:
            return "strong"
        if delta < -0.1:
            return "moderate"
        if delta <= 0.1:
            return "limited"
        return "regressing"

    @staticmethod
    def _describe_stability(stability: float) -> str:
        if stability > 0.75:
            return "high stability"
        if stability > 0.5:
            return "moderate stability"
        if stability > 0.25:
            return "variable stability"
        return "low stability"

    @staticmethod
    def _focus_hint(metrics: Dict[str, float]) -> str:
        if metrics["mastery_score"] > 0.6 and metrics["stability_score"] > 0.6:
            return "introduce more complex contexts"
        if metrics["frequency_trend"] > 0.2:
            return "reinforce fundamentals with spaced review"
        if metrics["error_rate_delta"] > 0:
            return "address recurring pitfalls before new material"
        return "continue current pace with targeted drills"

    @staticmethod
    def _cold_start_text() -> str:
        return "Initial observations collected. Not enough data yet to assess trends."

    # ----- Persistence to Qdrant -----
    def _upsert_qdrant(self, summary: LearningSummaryRow) -> None:
        vector = self.embedder.embed_single(summary.summary_text)
        payload = {
            "user_id": summary.user_id,
            "window_type": summary.window_type,
            "scope": summary.scope,
            "scope_key": summary.scope_key or "",
            "computed_at": summary.computed_at.isoformat(),
        }
        self.qdrant.upsert(
            collection_name="learning_summary_embeddings",
            points=[
                {
                    "id": summary.id,
                    "vector": vector,
                    "payload": payload,
                }
            ],
        )


def run_learning_summary_batch(
    sql_store: InMemorySQLStore,
    qdrant_store: QdrantStore,
    embedder: Optional[Embedder] = None,
    as_of: Optional[datetime] = None,
) -> None:
    batch = LearningSummaryBatch(sql_store, qdrant_store, embedder or Embedder())
    batch.run(as_of=as_of)

