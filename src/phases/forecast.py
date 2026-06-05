"""Phase 4: Predictive analytics and scenario narrative."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import ValidationError

from src.config import get_settings
from src.llm.client import get_client, get_deployment_name
from src.models.forecast import (
    ForecastNarrative,
    ForecastPoint,
    ForecastResult,
    PrescriptiveAction,
)

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
DATE_NAME_PATTERN = re.compile(r"date|time|timestamp|period", re.IGNORECASE)
TIME_COLUMN_BLOCKLIST = re.compile(r"overtime|lifetime|parttime|fulltime|timeslast", re.IGNORECASE)
TARGET_PRIORITY = (
    "amount",
    "revenue",
    "sales",
    "monthlyincome",
    "income",
    "qty",
    "quantity",
    "units",
    "stock",
    "inventory",
    "level",
    "count",
    "volume",
    "hike",
    "rate",
)
TARGET_NAME_PATTERN = re.compile(
    r"amount|revenue|sales|income|qty|quantity|units|stock|inventory|level|count|volume|hike|rate",
    re.IGNORECASE,
)
SNAPSHOT_DIM_PRIORITY = ("department", "region", "jobrole", "segment", "category", "warehouse")
HORIZON_DAYS = 30
MIN_HISTORY_POINTS = 5
NO_TIME_COLUMN_MESSAGE = (
    "No time column detected in cleaned data. Add a date/datetime field "
    "(e.g. order_date) to enable 30-day forecasting."
)


class ForecastSkipped(Exception):
    """Raised internally when forecasting cannot run; converted to skipped status."""


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]


def detect_time_column(df: pl.DataFrame) -> str | None:
    """Auto-detect a date/datetime column."""
    for col in df.columns:
        if TIME_COLUMN_BLOCKLIST.search(col):
            continue
        if df[col].dtype in DATE_DTYPES:
            return col
    for col in df.columns:
        if TIME_COLUMN_BLOCKLIST.search(col):
            continue
        if DATE_NAME_PATTERN.search(col) and df[col].dtype in (pl.Utf8, pl.Date, pl.Datetime):
            return col
    return None


def detect_target_column(df: pl.DataFrame, time_col: str | None) -> str | None:
    """Auto-detect forecast target: units, revenue, stock level, etc."""
    numeric = [c for c in _numeric_columns(df) if c != time_col]
    lower_map = {c.lower(): c for c in numeric}

    for preferred in TARGET_PRIORITY:
        if preferred in lower_map:
            return lower_map[preferred]

    for col in numeric:
        if TARGET_NAME_PATTERN.search(col):
            return col

    return numeric[0] if numeric else None


def _parse_time_column(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    series = df[time_col]
    if series.dtype in DATE_DTYPES:
        return df.with_columns(pl.col(time_col).cast(pl.Date).alias(time_col))
    return df.with_columns(
        pl.col(time_col)
        .cast(pl.Utf8, strict=False)
        .str.strptime(pl.Date, strict=False)
        .alias(time_col)
    )


def prepare_daily_series(
    df: pl.DataFrame,
    time_col: str,
    target_col: str,
) -> pl.DataFrame:
    """Aggregate target by day (sum for flow metrics)."""
    work = _parse_time_column(df, time_col)
    work = work.filter(pl.col(time_col).is_not_null(), pl.col(target_col).is_not_null())
    if work.height == 0:
        raise ForecastSkipped("No valid rows with both time and target values.")

    target_lower = target_col.lower()
    use_mean = any(k in target_lower for k in ("stock", "inventory", "level"))
    agg_expr = pl.col(target_col).cast(pl.Float64).mean() if use_mean else pl.col(target_col).cast(pl.Float64).sum()

    daily = (
        work.group_by(time_col)
        .agg(agg_expr.alias("y"))
        .sort(time_col)
        .rename({time_col: "ds"})
    )
    if daily.height < MIN_HISTORY_POINTS:
        raise ForecastSkipped(
            f"Need at least {MIN_HISTORY_POINTS} daily observations for forecasting; "
            f"found {daily.height}.",
        )
    return daily


def _points_from_frame(
    frame: pl.DataFrame,
    *,
    include_bounds: bool = False,
) -> list[ForecastPoint]:
    points: list[ForecastPoint] = []
    for row in frame.to_dicts():
        ds = row["ds"]
        if isinstance(ds, date):
            date_str = ds.isoformat()
        else:
            date_str = str(ds)[:10]
        points.append(
            ForecastPoint(
                date=date_str,
                value=round(float(row["yhat"] if "yhat" in row else row["y"]), 4),
                lower=round(float(row["yhat_lower"]), 4) if include_bounds and row.get("yhat_lower") is not None else None,
                upper=round(float(row["yhat_upper"]), 4) if include_bounds and row.get("yhat_upper") is not None else None,
            )
        )
    return points


def _fit_prophet(daily: pl.DataFrame, horizon_days: int) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    import pandas as pd
    from prophet import Prophet

    pdf = daily.to_pandas()
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    model = Prophet(interval_width=0.8, daily_seasonality=False, weekly_seasonality=len(pdf) >= 14)
    model.fit(pdf)
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    pred = model.predict(future)

    last_hist = pdf["ds"].max()
    future_pred = pred[pred["ds"] > last_hist][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    future_pl = pl.from_pandas(future_pred)
    hist_pl = daily
    return hist_pl, future_pl, "prophet"


def _fit_sklearn(daily: pl.DataFrame, horizon_days: int) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    pdf = daily.to_pandas()
    pdf["ds"] = pd.to_datetime(pdf["ds"])

    pdf = pdf.sort_values("ds")
    pdf["t"] = np.arange(len(pdf))
    X = pdf[["t"]].values
    y = pdf["y"].values
    reg = LinearRegression()
    reg.fit(X, y)
    residuals = y - reg.predict(X)
    std = float(np.std(residuals)) if len(residuals) > 1 else float(np.mean(np.abs(residuals)) or 1.0)

    last_date = pdf["ds"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon_days, freq="D")
    future_t = np.arange(len(pdf), len(pdf) + horizon_days).reshape(-1, 1)
    yhat = reg.predict(future_t)

    hist_pl = pl.DataFrame(
        {
            "ds": pdf["ds"].dt.date,
            "y": y,
        }
    )
    future_pl = pl.DataFrame(
        {
            "ds": [d.date() for d in future_dates],
            "yhat": yhat,
            "yhat_lower": yhat - 1.96 * std,
            "yhat_upper": yhat + 1.96 * std,
        }
    )
    return hist_pl, future_pl, "sklearn_linear"


def train_forecast(
    daily: pl.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[list[ForecastPoint], list[ForecastPoint], str]:
    """Train Prophet with sklearn fallback; return history and forecast points."""
    try:
        _hist, future_pl, model_name = _fit_prophet(daily, horizon_days)
        history = [
            ForecastPoint(date=str(r["ds"])[:10], value=round(float(r["y"]), 4))
            for r in daily.to_dicts()
        ]
        forecast = _points_from_frame(future_pl, include_bounds=True)
        return history, forecast, model_name
    except Exception:
        _hist, future_pl, model_name = _fit_sklearn(daily, horizon_days)
        history = [
            ForecastPoint(date=str(r["ds"])[:10], value=round(float(r["y"]), 4))
            for r in daily.to_dicts()
        ]
        forecast = _points_from_frame(future_pl, include_bounds=True)
        return history, forecast, model_name


def detect_snapshot_dimension(df: pl.DataFrame) -> str | None:
    """Pick a low-cardinality categorical column for cross-sectional analytics."""
    lower_map = {_normalize_col_key(c): c for c in df.columns}
    for preferred in SNAPSHOT_DIM_PRIORITY:
        if preferred in lower_map:
            col = lower_map[preferred]
            if 2 <= df[col].n_unique() <= 20:
                return col
    for col in df.columns:
        if df[col].dtype in (pl.Utf8, pl.Categorical, pl.String):
            n = df[col].n_unique()
            if 2 <= n <= 15:
                return col
    return None


def _normalize_col_key(name: str) -> str:
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def build_snapshot_forecast(df: pl.DataFrame) -> ForecastResult | None:
    """
    When no time column exists, produce cross-sectional 'forecast' (compare groups).

    Typical for HR/employee snapshots: avg monthly income by department, etc.
    """
    target_col = detect_target_column(df, None)
    dim_col = detect_snapshot_dimension(df)
    if not target_col or not dim_col:
        return None

    agg = (
        df.group_by(dim_col)
        .agg(pl.col(target_col).cast(pl.Float64, strict=False).mean().alias("y"))
        .sort("y", descending=True)
    )
    if agg.height < 2:
        return None

    points = [
        ForecastPoint(date=str(row[dim_col]), value=round(float(row["y"]), 2))
        for row in agg.to_dicts()
    ]
    top, bottom = points[0], points[-1]
    narrative = (
        f"No date column in this dataset — showing **snapshot analytics** instead of a time forecast. "
        f"Average **{target_col}** by **{dim_col}**: highest in **{top.date}** ({top.value:,.0f}), "
        f"lowest in **{bottom.date}** ({bottom.value:,.0f})."
    )
    actions = [
        PrescriptiveAction(
            action=f"Investigate drivers of {target_col} gap between {top.date} and {bottom.date}.",
            due_date=(date.today() + timedelta(days=14)).isoformat(),
        ),
        PrescriptiveAction(
            action=f"Set targets for underperforming {dim_col} groups using top quartile as benchmark.",
            due_date=(date.today() + timedelta(days=30)).isoformat(),
        ),
    ]
    return ForecastResult(
        status="snapshot",
        time_column=dim_col,
        target_column=target_col,
        model="group_mean_snapshot",
        horizon_days=0,
        history=points,
        forecast=points,
        forecast_narrative=narrative,
        prescriptive_actions=actions,
        narrative_source="heuristic",
        user_message=None,
    )


def render_snapshot_chart(
    points: list[ForecastPoint],
    output_path: Path,
    *,
    target_column: str,
    dimension_column: str,
) -> Path:
    """Bar chart for snapshot (no time series) analytics."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [p.date for p in points]
    values = [p.value for p in points]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2563eb" if i == 0 else "#64748b" for i in range(len(values))]
    ax.bar(labels, values, color=colors)
    ax.set_title(f"Average {target_column} by {dimension_column}")
    ax.set_xlabel(dimension_column)
    ax.set_ylabel(f"Mean {target_column}")
    plt.xticks(rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def render_forecast_chart(
    history: list[ForecastPoint],
    forecast: list[ForecastPoint],
    output_path: Path,
    *,
    target_column: str,
) -> Path:
    """Save forecast.png with history, forecast line, and confidence band."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist_dates = [datetime.fromisoformat(p.date) for p in history]
    hist_vals = [p.value for p in history]
    fc_dates = [datetime.fromisoformat(p.date) for p in forecast]
    fc_vals = [p.value for p in forecast]
    lowers = [p.lower if p.lower is not None else p.value for p in forecast]
    uppers = [p.upper if p.upper is not None else p.value for p in forecast]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_dates, hist_vals, "o-", color="#2563eb", label="History", linewidth=2)
    if hist_dates and fc_dates:
        ax.plot(
            [hist_dates[-1], fc_dates[0]],
            [hist_vals[-1], fc_vals[0]],
            color="#94a3b8",
            linestyle="--",
        )
    ax.plot(fc_dates, fc_vals, "o-", color="#dc2626", label="Forecast", linewidth=2)
    ax.fill_between(fc_dates, lowers, uppers, color="#fca5a5", alpha=0.35, label="80% interval")
    ax.set_title(f"30-day forecast — {target_column}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def build_heuristic_narrative(
    result: ForecastResult,
) -> tuple[str, list[PrescriptiveAction]]:
    """Template narrative when LLM is unavailable."""
    if not result.forecast:
        return "Forecast unavailable.", []

    last_hist = result.history[-1].value if result.history else 0.0
    end_fc = result.forecast[-1].value
    change_pct = ((end_fc - last_hist) / last_hist * 100) if last_hist else 0.0
    direction = "increase" if change_pct > 0 else "decrease"

    narrative = (
        f"Based on {result.model} over {len(result.history)} daily observations, "
        f"{result.target_column} is projected to {direction} by approximately "
        f"{abs(change_pct):.1f}% over the next {result.horizon_days} days "
        f"(from {last_hist:.2f} to {end_fc:.2f})."
    )

    start_action_date = (date.today() + timedelta(days=7)).isoformat()
    review_date = (date.today() + timedelta(days=14)).isoformat()
    actions = [
        PrescriptiveAction(
            action=f"Align inventory and staffing with projected {result.target_column} trend.",
            due_date=start_action_date,
        ),
        PrescriptiveAction(
            action="Review forecast vs actuals and adjust safety stock if deviation exceeds 10%.",
            due_date=review_date,
        ),
    ]
    return narrative, actions


def generate_forecast_narrative_llm(
    result: ForecastResult,
) -> tuple[str, list[PrescriptiveAction], Literal["llm", "heuristic"]]:
    """LLM generates forecast_narrative and two prescriptive actions with dates."""
    settings = get_settings()
    if not settings.azure_openai_api_key:
        narrative, actions = build_heuristic_narrative(result)
        return narrative, actions, "heuristic"

    compact = {
        "time_column": result.time_column,
        "target_column": result.target_column,
        "model": result.model,
        "horizon_days": result.horizon_days,
        "history_tail": [p.model_dump() for p in result.history[-7:]],
        "forecast_head": [p.model_dump() for p in result.forecast[:7]],
        "forecast_tail": [p.model_dump() for p in result.forecast[-7:]],
    }
    system = (
        "You are a business analyst. Write a concise forecast_narrative (2-4 sentences) "
        "and exactly 2 prescriptive_actions with due_date in YYYY-MM-DD format. "
        "Dates must be in the future relative to today."
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
            response_format=ForecastNarrative,
            temperature=0.2,
        )
        parsed = completion.choices[0].message.parsed
        if parsed:
            actions = parsed.prescriptive_actions[:2]
            return parsed.forecast_narrative, actions, "llm"
    except Exception:
        pass

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        parsed = ForecastNarrative.model_validate_json(response.choices[0].message.content or "{}")
        return parsed.forecast_narrative, parsed.prescriptive_actions[:2], "llm"
    except (ValidationError, Exception):
        narrative, actions = build_heuristic_narrative(result)
        return narrative, actions, "heuristic"


def run_forecast_pipeline(
    df: pl.DataFrame,
    *,
    use_llm: bool = True,
    horizon_days: int = HORIZON_DAYS,
) -> ForecastResult:
    """Run detection, training, and narrative generation."""
    time_col = detect_time_column(df)
    if time_col is not None and TIME_COLUMN_BLOCKLIST.search(time_col):
        time_col = None
    if time_col is None:
        snapshot = build_snapshot_forecast(df)
        if snapshot is not None:
            return snapshot
        return ForecastResult(
            status="skipped",
            user_message=NO_TIME_COLUMN_MESSAGE,
        )

    target_col = detect_target_column(df, time_col)
    if target_col is None:
        return ForecastResult(
            status="skipped",
            user_message="No numeric target column (amount, revenue, qty, stock) found for forecasting.",
            time_column=time_col,
        )

    try:
        daily = prepare_daily_series(df, time_col, target_col)
        history, forecast, model_name = train_forecast(daily, horizon_days=horizon_days)
    except ForecastSkipped as exc:
        return ForecastResult(
            status="skipped",
            user_message=str(exc),
            time_column=time_col,
            target_column=target_col,
        )
    except Exception as exc:
        return ForecastResult(
            status="failed",
            user_message=f"Forecast training failed: {exc}",
            time_column=time_col,
            target_column=target_col,
        )

    result = ForecastResult(
        status="success",
        time_column=time_col,
        target_column=target_col,
        model=model_name,
        horizon_days=horizon_days,
        history=history,
        forecast=forecast,
    )

    if use_llm:
        narrative, actions, source = generate_forecast_narrative_llm(result)
        result = result.model_copy(
            update={
                "forecast_narrative": narrative,
                "prescriptive_actions": actions,
                "narrative_source": source,
            },
        )
    else:
        narrative, actions = build_heuristic_narrative(result)
        result = result.model_copy(
            update={
                "forecast_narrative": narrative,
                "prescriptive_actions": actions,
                "narrative_source": "heuristic",
            },
        )

    return result


def run_forecast(
    cleaned_path: Path,
    artifact_dir: Path,
    *,
    use_llm: bool = True,
    horizon_days: int = HORIZON_DAYS,
) -> Path:
    """Execute forecasting; returns path to forecast.json."""
    cleaned_path = cleaned_path.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(cleaned_path)
    result = run_forecast_pipeline(df, use_llm=use_llm, horizon_days=horizon_days)

    json_path = artifact_dir / "forecast.json"
    chart_path = artifact_dir / "forecast.png"

    if result.status == "snapshot" and result.forecast:
        render_snapshot_chart(
            result.forecast,
            chart_path,
            target_column=result.target_column or "target",
            dimension_column=result.time_column or "group",
        )
        result = result.model_copy(update={"chart_path": str(chart_path.name)})
    elif result.status == "success" and result.forecast:
        render_forecast_chart(
            result.history,
            result.forecast,
            chart_path,
            target_column=result.target_column or "target",
        )
        result = result.model_copy(update={"chart_path": str(chart_path.name)})

    payload = result.model_dump(mode="json")
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def load_forecast(path: Path) -> ForecastResult:
    """Load forecast.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("generated_at", None)
    return ForecastResult.model_validate(data)
