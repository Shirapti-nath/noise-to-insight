"""Phase 2: Pattern discovery and insight ranking."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import duckdb
import polars as pl
from pydantic import ValidationError

from src.config import get_settings
from src.llm.client import get_client, get_deployment_name
from src.models.patterns import PatternDiscoveryResult, RankedInsights
from src.models.state import InsightCard

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}
DATE_DTYPES = {pl.Date, pl.Datetime}
DATE_NAME_PATTERN = re.compile(r"date|time|timestamp", re.IGNORECASE)
MAX_CATEGORIES = 25
MIN_ROWS = 3


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]


def _categorical_columns(df: pl.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if df[c].dtype in (pl.Utf8, pl.Categorical):
            n_unique = df[c].n_unique()
            if 2 <= n_unique <= MAX_CATEGORIES:
                cols.append(c)
    return cols


def _date_columns(df: pl.DataFrame) -> list[str]:
    cols = [c for c in df.columns if df[c].dtype in DATE_DTYPES]
    for c in df.columns:
        if c not in cols and DATE_NAME_PATTERN.search(c) and df[c].dtype in (pl.Utf8, pl.Date, pl.Datetime):
            cols.append(c)
    return cols


def compute_correlations(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Pairwise Pearson correlations for numeric columns."""
    num_cols = _numeric_columns(df)
    if len(num_cols) < 2 or df.height < MIN_ROWS:
        return []

    corr_df = df.select(num_cols).corr()
    row_index = {name: idx for idx, name in enumerate(num_cols)}
    patterns: list[dict[str, Any]] = []
    for i, col_a in enumerate(num_cols):
        for col_b in num_cols[i + 1 :]:
            # Polars 1.x: square matrix (no "column" index column)
            if "column" in corr_df.columns:
                value = corr_df[col_a][corr_df["column"] == col_b]
                if value is None or len(value) == 0:
                    continue
                r = float(value[0])
            else:
                r = float(corr_df[col_b][row_index[col_a]])
            if r != r:  # NaN
                continue
            if abs(r) < 0.3:
                continue
            patterns.append(
                {
                    "type": "correlation",
                    "score": round(abs(r), 4),
                    "columns": [col_a, col_b],
                    "correlation": round(r, 4),
                    "direction": "positive" if r > 0 else "negative",
                    "description": f"{col_a} and {col_b} correlation r={r:.2f}",
                }
            )
    return sorted(patterns, key=lambda p: p["score"], reverse=True)[:10]


def compute_segment_lifts(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Compare segment means vs global mean for categorical × numeric pairs."""
    cat_cols = _categorical_columns(df)[:6]
    num_cols = _numeric_columns(df)[:4]
    patterns: list[dict[str, Any]] = []

    for cat in cat_cols:
        for num in num_cols:
            if cat == num:
                continue
            overall = df[num].mean()
            if overall is None or overall == 0:
                continue

            grouped = (
                df.group_by(cat)
                .agg(
                    pl.col(num).mean().alias("segment_mean"),
                    pl.len().alias("count"),
                )
                .filter(pl.col("count") >= 1)
            )
            if grouped.height < 2:
                continue

            rows = grouped.to_dicts()
            lifts = [
                (
                    row,
                    (float(row["segment_mean"]) / float(overall)) - 1.0,
                )
                for row in rows
            ]
            max_abs = max(abs(lift) for _, lift in lifts)
            tied = [
                (row, lift)
                for row, lift in lifts
                if math.isclose(abs(lift), max_abs, rel_tol=1e-6, abs_tol=1e-6)
            ]
            best_row, best_lift = max(tied, key=lambda item: float(item[0]["segment_mean"]))

            if abs(best_lift) < 0.05:
                continue

            patterns.append(
                {
                    "type": "segment_lift",
                    "score": round(min(abs(best_lift), 5.0), 4),
                    "segment_column": cat,
                    "metric_column": num,
                    "segment_value": str(best_row[cat]),
                    "segment_mean": round(float(best_row["segment_mean"]), 4),
                    "global_mean": round(float(overall), 4),
                    "lift_pct": round(best_lift * 100, 2),
                    "count": int(best_row["count"]),
                    "description": (
                        f"{cat}={best_row[cat]} has {num} mean "
                        f"{best_row['segment_mean']:.2f} vs global {overall:.2f} "
                        f"({best_lift * 100:+.1f}%)"
                    ),
                }
            )

    return sorted(patterns, key=lambda p: p["score"], reverse=True)[:12]


def compute_time_trends(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Weekly aggregation and trend direction for date + numeric columns."""
    date_cols = _date_columns(df)
    num_cols = _numeric_columns(df)
    patterns: list[dict[str, Any]] = []

    for date_col in date_cols[:2]:
        series = df[date_col]
        if series.dtype == pl.Utf8:
            parsed = pl.col(date_col).str.strptime(pl.Date, strict=False)
            work = df.with_columns(parsed.alias("_dt")).filter(pl.col("_dt").is_not_null())
            dt_col = "_dt"
        else:
            work = df.filter(pl.col(date_col).is_not_null())
            dt_col = date_col

        if work.height < MIN_ROWS:
            continue

        metric = num_cols[0] if num_cols else None
        weekly = work.with_columns(pl.col(dt_col).dt.truncate("1w").alias("_week")).group_by(
            "_week"
        )
        if metric:
            weekly_df = weekly.agg(
                pl.col(metric).sum().alias("metric_sum"),
                pl.len().alias("count"),
            ).sort("_week")
        else:
            weekly_df = weekly.agg(pl.len().alias("count")).sort("_week")

        if weekly_df.height < 2:
            continue

        values = (
            weekly_df["metric_sum"].to_list()
            if metric and "metric_sum" in weekly_df.columns
            else weekly_df["count"].to_list()
        )
        first_val = float(values[0])
        last_val = float(values[-1])
        change_pct = ((last_val - first_val) / first_val * 100) if first_val else 0.0
        direction = "up" if change_pct > 5 else "down" if change_pct < -5 else "flat"

        patterns.append(
            {
                "type": "time_trend",
                "score": round(min(abs(change_pct) / 100, 3.0), 4),
                "date_column": date_col,
                "metric_column": metric or "row_count",
                "periods": weekly_df.height,
                "first_value": round(first_val, 4),
                "last_value": round(last_val, 4),
                "change_pct": round(change_pct, 2),
                "direction": direction,
                "description": (
                    f"Weekly {metric or 'volume'} trend {direction}: "
                    f"{change_pct:+.1f}% from first to last period"
                ),
            }
        )

    return patterns


def compute_top_categorical_combos(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Most frequent two-way categorical combinations."""
    cat_cols = _categorical_columns(df)
    if len(cat_cols) < 2:
        return []

    patterns: list[dict[str, Any]] = []
    for i, col_a in enumerate(cat_cols[:4]):
        for col_b in cat_cols[i + 1 : 4]:
            combo = (
                df.group_by(col_a, col_b)
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(5)
            )
            if combo.height == 0:
                continue
            top = combo.head(1).to_dicts()[0]
            share = top["count"] / df.height if df.height else 0
            patterns.append(
                {
                    "type": "categorical_combo",
                    "score": round(share, 4),
                    "columns": [col_a, col_b],
                    "values": {col_a: str(top[col_a]), col_b: str(top[col_b])},
                    "count": int(top["count"]),
                    "share_pct": round(share * 100, 2),
                    "description": (
                        f"Top combo {col_a}={top[col_a]} + {col_b}={top[col_b]} "
                        f"appears in {share * 100:.1f}% of rows"
                    ),
                }
            )
    return sorted(patterns, key=lambda p: p["score"], reverse=True)[:8]


def compute_statistics_duckdb(df: pl.DataFrame) -> dict[str, Any]:
    """Summary stats via DuckDB for compact LLM context."""
    num_cols = _numeric_columns(df)
    if not num_cols:
        return {"numeric_summary": {}}

    conn = duckdb.connect()
    try:
        conn.register("cleaned", df)
        parts: list[str] = []
        for col in num_cols[:8]:
            quoted = col.replace('"', '""')
            parts.append(f'AVG("{quoted}") AS avg_{col}')
            parts.append(f'MIN("{quoted}") AS min_{col}')
            parts.append(f'MAX("{quoted}") AS max_{col}')
        cols_sql = ", ".join(parts)
        cursor = conn.execute(f"SELECT {cols_sql} FROM cleaned")
        row = cursor.fetchone()
        summary = {
            desc[0]: round(float(val), 4) if val is not None else None
            for desc, val in zip(cursor.description, row, strict=True)
        }
        return {"numeric_summary": summary, "row_count": df.height, "column_count": df.width}
    finally:
        conn.close()


def discover_raw_patterns(df: pl.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all statistical detectors and merge ranked raw patterns."""
    statistics = compute_statistics_duckdb(df)
    statistics["numeric_columns"] = _numeric_columns(df)
    statistics["categorical_columns"] = _categorical_columns(df)
    statistics["date_columns"] = _date_columns(df)

    buckets = [
        compute_correlations(df),
        compute_segment_lifts(df),
        compute_time_trends(df),
        compute_top_categorical_combos(df),
    ]
    raw_patterns: list[dict[str, Any]] = []
    for bucket in buckets:
        raw_patterns.extend(bucket)

    raw_patterns.sort(key=lambda p: p.get("score", 0), reverse=True)
    statistics["pattern_counts"] = {
        "correlation": sum(1 for p in raw_patterns if p["type"] == "correlation"),
        "segment_lift": sum(1 for p in raw_patterns if p["type"] == "segment_lift"),
        "time_trend": sum(1 for p in raw_patterns if p["type"] == "time_trend"),
        "categorical_combo": sum(1 for p in raw_patterns if p["type"] == "categorical_combo"),
    }
    return statistics, raw_patterns


def _pattern_to_insight(pattern: dict[str, Any], rank: int) -> InsightCard:
    """Map a statistical pattern to an InsightCard (heuristic path)."""
    ptype = pattern.get("type", "pattern")
    impact = "medium"
    score = pattern.get("score", 0)
    if score >= 1.0 or abs(pattern.get("lift_pct", 0)) >= 30:
        impact = "high"
    elif score < 0.15:
        impact = "low"

    title = {
        "correlation": f"Strong link: {pattern.get('columns', ['', ''])[0]} ↔ {pattern.get('columns', ['', ''])[1]}",
        "segment_lift": f"{pattern.get('segment_column')} driver: {pattern.get('segment_value')}",
        "time_trend": f"Trend {pattern.get('direction', 'shift')} on {pattern.get('metric_column', 'volume')}",
        "categorical_combo": f"Dominant pair: {pattern.get('columns', ['', ''])[0]} × {pattern.get('columns', ['', ''])[1]}",
    }.get(ptype, f"Pattern: {ptype}")

    return InsightCard(
        title=title,
        summary=pattern.get("description", ""),
        impact=impact,
        evidence={k: v for k, v in pattern.items() if k not in ("description",)},
    )


def build_heuristic_insights(
    raw_patterns: list[dict[str, Any]],
    *,
    max_insights: int = 5,
) -> list[InsightCard]:
    """Rank insights from statistical patterns without LLM."""
    if not raw_patterns:
        return [
            InsightCard(
                title="Limited signal in dataset",
                summary="Not enough variation to surface strong patterns after cleaning.",
                impact="low",
                evidence={"row_patterns": 0},
            )
        ]
    return [_pattern_to_insight(p, i) for i, p in enumerate(raw_patterns[:max_insights])]


def rank_insights_llm(
    statistics: dict[str, Any],
    raw_patterns: list[dict[str, Any]],
    *,
    max_insights: int = 5,
) -> tuple[list[InsightCard], Literal["llm", "heuristic"]]:
    """Use Azure OpenAI to rank patterns into business InsightCards."""
    settings = get_settings()
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        return build_heuristic_insights(raw_patterns, max_insights=max_insights), "heuristic"

    compact = {
        "statistics": statistics,
        "raw_patterns": raw_patterns[:20],
    }
    system = (
        "You are a senior data analyst. Given statistical patterns from cleaned operational data, "
        "produce 3-5 InsightCards ranked by business impact. Each insight needs a clear title, "
        "actionable summary (so what), impact level (high/medium/low), and evidence citing "
        "column names and metric values from the input."
    )
    user = (
        f"Statistical findings:\n{json.dumps(compact, indent=2, default=str)}\n\n"
        f"Return up to {max_insights} insights ranked by importance."
    )

    client = get_client()
    deployment = get_deployment_name()

    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=RankedInsights,
            temperature=0.2,
        )
        parsed = completion.choices[0].message.parsed
        if parsed and parsed.insights:
            return parsed.insights[:max_insights], "llm"
    except Exception:
        pass

    try:
        schema = RankedInsights.model_json_schema()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system + f"\nSchema:\n{json.dumps(schema)}"},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        ranked = RankedInsights.model_validate_json(content)
        return ranked.insights[:max_insights], "llm"
    except (ValidationError, Exception):
        return build_heuristic_insights(raw_patterns, max_insights=max_insights), "heuristic"


def run_pattern_discovery(
    df: pl.DataFrame,
    *,
    use_llm: bool = True,
    insights: list[InsightCard] | None = None,
    insight_source: Literal["llm", "heuristic", "provided"] | None = None,
) -> PatternDiscoveryResult:
    """Discover patterns and rank insights from a cleaned DataFrame."""
    statistics, raw_patterns = discover_raw_patterns(df)

    if insights is None:
        if use_llm:
            insights, source = rank_insights_llm(statistics, raw_patterns)
        else:
            insights = build_heuristic_insights(raw_patterns)
            source = "heuristic"
    else:
        source = insight_source or "provided"

    return PatternDiscoveryResult(
        row_count=df.height,
        statistics=statistics,
        raw_patterns=raw_patterns,
        insights=insights,
        insight_source=source,
    )


def run_patterns(
    cleaned_path: Path,
    artifact_dir: Path,
    *,
    use_llm: bool = True,
    insights: list[InsightCard] | None = None,
    insight_source: Literal["llm", "heuristic", "provided"] | None = None,
) -> Path:
    """
    Execute pattern discovery phase.

    Reads cleaned.parquet, writes patterns.json.
    """
    cleaned_path = cleaned_path.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(cleaned_path)
    result = run_pattern_discovery(
        df,
        use_llm=use_llm,
        insights=insights,
        insight_source=insight_source,
    )

    output_path = artifact_dir / "patterns.json"
    payload = result.model_dump(mode="json")
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_patterns(patterns_path: Path) -> PatternDiscoveryResult:
    """Load patterns.json from disk."""
    data = json.loads(patterns_path.read_text(encoding="utf-8"))
    data.pop("generated_at", None)
    return PatternDiscoveryResult.model_validate(data)


def get_top_insights(
    patterns_path: Path | PatternDiscoveryResult,
    n: int = 3,
) -> list[InsightCard]:
    """Return the top n insight cards for Streamlit / wow panel."""
    result = (
        patterns_path
        if isinstance(patterns_path, PatternDiscoveryResult)
        else load_patterns(patterns_path)
    )
    return result.insights[:n]
