"""Phase 3: Anomaly detection and LLM explanations."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import get_settings
from src.llm.client import get_client, get_deployment_name
from src.models.anomalies import AnomalyDetectionResult, ExplainedAnomalies
from src.models.state import AnomalyRecord

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
TEXT_COLUMN_PATTERN = re.compile(
    r"reason|comment|description|note|text|message|support|supplier|issue|delay",
    re.IGNORECASE,
)
ENTITY_COLUMN_PRIORITY = (
    "employee_number",
    "employeenumber",
    "employee_id",
    "emp_id",
    "supplier_id",
    "order_id",
    "sku",
    "warehouse_id",
    "region",
    "department",
    "jobrole",
    "customer_id",
    "product_id",
    "entity",
    "id",
)


def _normalize_col_key(name: str) -> str:
    return re.sub(r"[\s-]+", "_", name.strip().lower())
MIN_ROWS_FOR_IF = 5
DEFAULT_TOP_N = 10


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]


def _text_columns(df: pl.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if df[c].dtype in (pl.Utf8, pl.Categorical) and TEXT_COLUMN_PATTERN.search(c):
            cols.append(c)
    return cols


def _entity_column(df: pl.DataFrame) -> str | None:
    lower_map = {_normalize_col_key(c): c for c in df.columns}
    for preferred in ENTITY_COLUMN_PRIORITY:
        if preferred in lower_map:
            return lower_map[preferred]
    for key, col in lower_map.items():
        if key.endswith("_id") or key.endswith("_number") or key in ("id", "name"):
            return col
    # Fallback: highest-cardinality column (likely employee / record id)
    best_col: str | None = None
    best_unique = 0
    for col in df.columns:
        n_unique = df[col].n_unique()
        if n_unique > best_unique and n_unique > 1:
            best_unique = n_unique
            best_col = col
    return best_col


def _entity_label_for_row(df: pl.DataFrame, row_idx: int, entity_col: str | None) -> tuple[str, str | None]:
    """Return human-readable entity label and column name for a row."""
    if entity_col is not None:
        val = df[entity_col][row_idx]
        if val is not None and str(val).strip():
            return str(val).strip(), entity_col
    return f"Row {row_idx + 1}", entity_col


def build_graph_entity_id(entity_column: str | None, entity_value: str) -> str:
    """Stable node id for knowledge graph linking."""
    column = (entity_column or "entity").lower().replace(" ", "_")
    prefix = column.removesuffix("_id") if column.endswith("_id") else column
    safe_value = re.sub(r"[^\w.-]+", "_", str(entity_value).strip())[:64]
    return f"{prefix}:{safe_value}"


def link_anomaly_to_graph_entity(
    record: AnomalyRecord,
    *,
    entity_column: str | None,
    entity_value: str | None,
) -> AnomalyRecord:
    """Attach graph_entity_id to an anomaly record for Phase 5 viz."""
    if record.graph_entity_id:
        return record
    value = entity_value or record.entity
    graph_id = build_graph_entity_id(entity_column, value)
    return record.model_copy(update={"graph_entity_id": graph_id})


def build_entity_index(anomalies: list[AnomalyRecord]) -> dict[str, list[int]]:
    """Map graph_entity_id -> anomaly list indices."""
    index: dict[str, list[int]] = {}
    for i, record in enumerate(anomalies):
        if not record.graph_entity_id:
            continue
        index.setdefault(record.graph_entity_id, []).append(i)
    return index


def engineer_features(df: pl.DataFrame) -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """
    Build numeric feature matrix: median-imputed numeric cols + scaled values.

    Returns (X, feature_names, feature_frame aligned to df rows).
    """
    num_cols = _numeric_columns(df)
    if not num_cols:
        raise ValueError("No numeric columns available for anomaly detection")

    feature_frame = df.select(num_cols).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(pl.col(c).median()) for c in num_cols]
    )
    matrix = feature_frame.to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, num_cols, feature_frame


def _severity_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def detect_numeric_anomalies(
    df: pl.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    contamination: float | str = "auto",
) -> list[dict[str, Any]]:
    """IsolationForest on engineered features; return raw anomaly candidates."""
    if df.height < MIN_ROWS_FOR_IF:
        return []

    features, feature_names, feature_frame = engineer_features(df)
    entity_col = _entity_column(df)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    raw_scores = model.score_samples(features)
    decision = model.decision_function(features)

    # Higher anomaly_score = more anomalous (invert sklearn score_samples)
    score_min, score_max = float(raw_scores.min()), float(raw_scores.max())
    span = score_max - score_min or 1.0
    anomaly_scores = 1.0 - ((raw_scores - score_min) / span)

    row_indices = np.argsort(anomaly_scores)[::-1][:top_n]
    candidates: list[dict[str, Any]] = []

    for rank, idx in enumerate(row_indices):
        row_idx = int(idx)
        row_vec = features[row_idx]
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        z_scores = np.abs((row_vec - means) / stds)
        max_idx = int(np.argmax(z_scores))
        metric = feature_names[max_idx]
        deviations = {
            feature_names[i]: round(float(row_vec[i]), 4) for i in range(len(feature_names))
        }

        entity_value, entity_col = _entity_label_for_row(df, row_idx, entity_col)

        norm_score = float(anomaly_scores[row_idx])
        candidates.append(
            {
                "source": "isolation_forest",
                "row_index": row_idx,
                "entity": entity_value,
                "entity_column": entity_col,
                "metric": metric,
                "score": round(min(max(norm_score, 0.0), 1.0), 4),
                "severity": _severity_from_score(norm_score),
                "evidence": {
                    "anomaly_score": round(norm_score, 4),
                    "decision_function": round(float(decision[row_idx]), 4),
                    "feature_values": {k: round(v, 4) for k, v in deviations.items()},
                    "rank": rank + 1,
                    "row_index": row_idx,
                },
            }
        )

    return candidates


def _text_length_outliers(df: pl.DataFrame, col: str) -> list[dict[str, Any]]:
    """Heuristic text outliers by length z-score."""
    lengths = df[col].cast(pl.Utf8, strict=False).str.len_chars().fill_null(0)
    work = df.with_columns(lengths.alias("_txt_len"))
    mean_len = work["_txt_len"].mean() or 0
    std_len = work["_txt_len"].std() or 1.0
    if std_len == 0:
        return []

    entity_col = _entity_column(df)
    outliers: list[dict[str, Any]] = []
    for row_idx in range(work.height):
        length = int(work["_txt_len"][row_idx])
        z = abs((length - mean_len) / std_len)
        if z < 1.5:
            continue
        text_val = str(work[col][row_idx] or "")
        entity_value, entity_col = _entity_label_for_row(work, row_idx, entity_col)
        outliers.append(
            {
                "source": "text_heuristic",
                "row_index": row_idx,
                "entity": entity_value,
                "entity_column": entity_col,
                "metric": col,
                "score": round(min(z / 4.0, 1.0), 4),
                "severity": _severity_from_score(min(z / 4.0, 1.0)),
                "evidence": {"text": text_val[:240], "length_z": round(z, 2)},
            }
        )
    return sorted(outliers, key=lambda x: x["score"], reverse=True)[:5]


def detect_text_anomalies(
    df: pl.DataFrame,
    *,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """Flag unusual text in support/supplier columns (heuristic + optional LLM)."""
    text_cols = _text_columns(df)
    if not text_cols:
        return []

    heuristic: list[dict[str, Any]] = []
    for col in text_cols[:2]:
        heuristic.extend(_text_length_outliers(df, col))

    settings = get_settings()
    if not use_llm or not settings.azure_openai_api_key or df.height < 3:
        return heuristic

    return _detect_text_anomalies_llm(df, text_cols, heuristic)


def _detect_text_anomalies_llm(
    df: pl.DataFrame,
    text_cols: list[str],
    heuristic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ask LLM to flag semantically unusual text entries."""
    entity_col = _entity_column(df)
    samples: list[dict[str, Any]] = []
    for col in text_cols[:2]:
        for row_idx in range(min(df.height, 30)):
            text = df[col][row_idx]
            if text is None or str(text).strip() == "":
                continue
            entity_value = str(df[entity_col][row_idx]) if entity_col else f"row_{row_idx}"
            samples.append(
                {
                    "row_index": row_idx,
                    "entity": entity_value,
                    "column": col,
                    "text": str(text)[:300],
                }
            )

    if not samples:
        return heuristic

    client = get_client()
    deployment = get_deployment_name()
    prompt = {
        "instruction": (
            "Identify up to 5 semantically anomalous text entries (complaints, delays, "
            "supplier issues) that differ from the majority tone or topic."
        ),
        "samples": samples[:25],
    }

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON: {\"flags\": [{\"row_index\": int, \"entity\": str, "
                        "\"column\": str, \"score\": 0-1, \"hypothesis\": str, "
                        "\"recommended_action\": str}]}"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        llm_flags = payload.get("flags", [])
        results: list[dict[str, Any]] = list(heuristic)
        for flag in llm_flags:
            row_idx = int(flag.get("row_index", -1))
            if row_idx < 0 or row_idx >= df.height:
                continue
            col = flag.get("column", text_cols[0])
            results.append(
                {
                    "source": "text_llm",
                    "row_index": row_idx,
                    "entity": str(flag.get("entity", "unknown")),
                    "entity_column": entity_col,
                    "metric": col,
                    "score": round(float(flag.get("score", 0.7)), 4),
                    "severity": _severity_from_score(float(flag.get("score", 0.7))),
                    "evidence": {"text": str(df[col][row_idx])[:240]},
                    "hypothesis": flag.get("hypothesis"),
                    "recommended_action": flag.get("recommended_action"),
                }
            )
        return results
    except Exception:
        return heuristic


def _find_matching_candidate(
    record: AnomalyRecord,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Match an explained record back to its statistical candidate."""
    for cand in candidates:
        if cand.get("metric") == record.metric and abs(float(cand.get("score", 0)) - record.score) < 0.02:
            return cand
    for cand in candidates:
        if cand.get("metric") == record.metric:
            return cand
    rank = record.evidence.get("rank")
    if rank is not None:
        for cand in candidates:
            if cand.get("evidence", {}).get("rank") == rank:
                return cand
    return {}


def _resolve_entity(
    record: AnomalyRecord,
    candidate: dict[str, Any],
    df: pl.DataFrame,
) -> tuple[str, str | None]:
    """Prefer statistical candidate entity; never leave employee IDs as unknown."""
    entity_col = candidate.get("entity_column") or _entity_column(df)
    entity = str(candidate.get("entity") or record.entity or "").strip()
    row_idx = candidate.get("row_index")
    if row_idx is None:
        row_idx = record.evidence.get("row_index")
    if (not entity or entity.lower() == "unknown") and row_idx is not None:
        entity, entity_col = _entity_label_for_row(df, int(row_idx), entity_col)
    if not entity or entity.lower() == "unknown":
        entity = f"Row {(int(row_idx) + 1) if row_idx is not None else '?'}"
    return entity, entity_col


def align_anomaly_record(
    record: AnomalyRecord,
    candidates: list[dict[str, Any]],
    df: pl.DataFrame,
) -> AnomalyRecord:
    """Restore entity labels and refresh hypothesis text after LLM enrichment."""
    cand = _find_matching_candidate(record, candidates)
    entity, entity_col = _resolve_entity(record, cand, df)
    hypothesis = record.hypothesis or ""
    action = record.recommended_action or ""
    if "unknown" in hypothesis.lower():
        hypothesis = hypothesis.replace("unknown", entity).replace("Unknown", entity)
    if "unknown" in action.lower():
        action = action.replace("unknown", entity).replace("Unknown", entity)
    evidence = {**record.evidence, **cand.get("evidence", {})}
    if cand.get("row_index") is not None:
        evidence["row_index"] = cand["row_index"]
    updated = record.model_copy(
        update={
            "entity": entity,
            "hypothesis": hypothesis,
            "recommended_action": action,
            "evidence": evidence,
        },
    )
    return link_anomaly_to_graph_entity(
        updated,
        entity_column=entity_col,
        entity_value=entity,
    )


def _candidate_to_record(candidate: dict[str, Any]) -> AnomalyRecord:
    score = float(candidate.get("score", 0.0))
    evidence = dict(candidate.get("evidence", {}))
    if candidate.get("row_index") is not None:
        evidence["row_index"] = candidate["row_index"]
    record = AnomalyRecord(
        entity=str(candidate.get("entity", "unknown")),
        metric=str(candidate.get("metric", "unknown")),
        score=score,
        severity=str(candidate.get("severity", _severity_from_score(score))),
        hypothesis=candidate.get("hypothesis"),
        recommended_action=candidate.get("recommended_action"),
        source=str(candidate.get("source", "isolation_forest")),
        evidence=evidence,
    )
    return link_anomaly_to_graph_entity(
        record,
        entity_column=candidate.get("entity_column"),
        entity_value=str(candidate.get("entity")),
    )


def build_heuristic_explanations(candidates: list[dict[str, Any]]) -> list[AnomalyRecord]:
    """Template explanations when LLM is unavailable."""
    records: list[AnomalyRecord] = []
    for cand in candidates:
        metric = cand.get("metric", "value")
        entity = cand.get("entity", "unknown")
        source = cand.get("source", "isolation_forest")
        if source.startswith("text"):
            hypothesis = (
                cand.get("hypothesis")
                or f"Unusual text pattern detected in {metric} for {entity}."
            )
            action = cand.get("recommended_action") or f"Review {metric} entry for {entity}."
        else:
            hypothesis = (
                f"Multivariate outlier: {metric} deviates from peer behavior for {entity}."
            )
            action = f"Investigate {entity} — validate {metric} and related operational events."
        enriched = {**cand, "hypothesis": hypothesis, "recommended_action": action}
        records.append(_candidate_to_record(enriched))
    return records


def explain_anomalies_llm(
    candidates: list[dict[str, Any]],
    df: pl.DataFrame,
) -> tuple[list[AnomalyRecord], Literal["llm", "heuristic"]]:
    """Merge statistical and text candidates into explained AnomalyRecords."""
    settings = get_settings()
    if not settings.azure_openai_api_key or not candidates:
        return build_heuristic_explanations(candidates), "heuristic"

    compact = {
        "row_count": df.height,
        "columns": df.columns,
        "candidates": candidates[:15],
    }
    system = (
        "You are an operations analyst. Enrich each anomaly candidate with a clear hypothesis "
        "and recommended_action. Preserve entity, metric, score; set severity "
        "(critical/high/medium/low) from score. Include graph_entity_id as "
        "'{entity_prefix}:{entity_value}' using entity_column when provided."
    )

    client = get_client()
    deployment = get_deployment_name()

    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
            response_format=ExplainedAnomalies,
            temperature=0.1,
        )
        parsed = completion.choices[0].message.parsed
        if parsed and parsed.anomalies:
            linked = [
                align_anomaly_record(a, candidates, df) for a in parsed.anomalies
            ]
            return linked, "llm"
    except Exception:
        pass

    try:
        schema = ExplainedAnomalies.model_json_schema()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system + f"\nSchema:\n{json.dumps(schema)}"},
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        explained = ExplainedAnomalies.model_validate_json(response.choices[0].message.content or "{}")
        linked = [align_anomaly_record(a, candidates, df) for a in explained.anomalies]
        return linked, "llm"
    except (ValidationError, Exception):
        return build_heuristic_explanations(candidates), "heuristic"


def discover_anomalies(
    df: pl.DataFrame,
    *,
    use_llm: bool = True,
    top_n: int = DEFAULT_TOP_N,
) -> AnomalyDetectionResult:
    """Run full anomaly detection pipeline on cleaned data."""
    numeric_candidates: list[dict[str, Any]] = []
    if _numeric_columns(df):
        try:
            numeric_candidates = detect_numeric_anomalies(df, top_n=top_n)
        except ValueError:
            numeric_candidates = []
    text_candidates = detect_text_anomalies(df, use_llm=use_llm)

    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cand in numeric_candidates + text_candidates:
        key = (
            str(cand.get("entity")),
            str(cand.get("metric")),
            int(cand.get("row_index", -1)),
        )
        existing = merged.get(key)
        if existing is None or cand.get("score", 0) > existing.get("score", 0):
            merged[key] = cand

    candidates = sorted(merged.values(), key=lambda c: c.get("score", 0), reverse=True)[:top_n]

    if use_llm:
        records, source = explain_anomalies_llm(candidates, df)
    else:
        records = build_heuristic_explanations(candidates)
        source = "heuristic"

    records = [align_anomaly_record(r, candidates, df) for r in records]
    records = sorted(records, key=lambda r: r.score, reverse=True)
    feature_cols = _numeric_columns(df)

    return AnomalyDetectionResult(
        row_count=df.height,
        feature_columns=feature_cols,
        anomalies=records,
        entity_index=build_entity_index(records),
        explanation_source=source,
    )


def run_anomalies(
    cleaned_path: Path,
    artifact_dir: Path,
    *,
    use_llm: bool = True,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    """Execute anomaly detection; returns path to anomalies.json."""
    cleaned_path = cleaned_path.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(cleaned_path)
    result = discover_anomalies(df, use_llm=use_llm, top_n=top_n)

    output_path = artifact_dir / "anomalies.json"
    payload = result.model_dump(mode="json")
    payload["anomaly_count"] = len(result.anomalies)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_anomalies(path: Path) -> AnomalyDetectionResult:
    """Load anomalies.json from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("generated_at", None)
    return AnomalyDetectionResult.model_validate(data)


def get_anomalies_for_entity(
    result: AnomalyDetectionResult | Path,
    graph_entity_id: str,
) -> list[AnomalyRecord]:
    """Return anomalies linked to a graph node id (for Phase 5 highlight)."""
    if isinstance(result, Path):
        result = load_anomalies(result)
    indices = result.entity_index.get(graph_entity_id, [])
    return [result.anomalies[i] for i in indices if i < len(result.anomalies)]
