from __future__ import annotations

import datetime
import logging

import pandas as pd


logger = logging.getLogger(__name__)


_ABSENCE_REQUIRED_COLUMNS = {
    "employee_id",
    "date_from",
    "date_to",
    "type",
}

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "si", "sì"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}


def _rename_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with legacy column names normalised."""

    rename_map = {}
    if "start_date" in df.columns and "date_from" not in df.columns:
        rename_map["start_date"] = "date_from"
    if "end_date" in df.columns and "date_to" not in df.columns:
        rename_map["end_date"] = "date_to"
    if "tipo" in df.columns and "type" not in df.columns:
        rename_map["tipo"] = "type"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _validate_absence_dates(df: pd.DataFrame) -> None:
    if (df["date_from"] > df["date_to"]).any():
        bad_rows = df.loc[df["date_from"] > df["date_to"], ["employee_id", "date_from", "date_to"]]
        raise ValueError(
            "Intervallo di assenza non valido: date_from deve essere <= date_to. "
            f"Righe: {bad_rows.to_dict(orient='records')}"
        )


def _coerce_optional_bool(series: pd.Series, default: bool) -> pd.Series:
    """Return a boolean Series parsed from ``series`` with ``default`` fallback."""

    result = []
    for value in series:
        if isinstance(value, (bool, int)) and value in (0, 1, True, False):
            result.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in _TRUE_VALUES:
            result.append(True)
        elif text in _FALSE_VALUES:
            result.append(False)
        elif text == "":
            result.append(default)
        else:
            result.append(default)
    return pd.Series(result, index=series.index, dtype=bool)


def load_absences(path: str) -> pd.DataFrame:
    """Load and normalise an absences CSV file."""

    df = pd.read_csv(path, dtype=str).fillna("")
    df = _rename_legacy_columns(df)

    missing = _ABSENCE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Il file di assenze deve contenere le colonne: "
            f"{sorted(_ABSENCE_REQUIRED_COLUMNS)}; mancanti: {sorted(missing)}"
        )

    normalized = df.loc[:, ["employee_id", "date_from", "date_to", "type"]].copy()
    normalized["employee_id"] = normalized["employee_id"].astype(str).str.strip()
    if normalized["employee_id"].eq("").any():
        raise ValueError("employee_id non può essere vuoto nelle assenze")

    for column in ("date_from", "date_to"):
        normalized[column] = (
            pd.to_datetime(normalized[column], format="%Y-%m-%d", errors="raise").dt.date
        )

    normalized["type"] = normalized["type"].astype(str).str.strip().str.upper()

    planned_series: pd.Series | None = None
    if "is_planned" in df.columns:
        planned_series = _coerce_optional_bool(df["is_planned"], default=True)
    elif "planned" in df.columns:
        planned_series = _coerce_optional_bool(df["planned"], default=True)
    elif "is_unplanned" in df.columns:
        planned_series = ~_coerce_optional_bool(df["is_unplanned"], default=False)
    elif "unplanned" in df.columns:
        planned_series = ~_coerce_optional_bool(df["unplanned"], default=False)

    if planned_series is None:
        normalized["is_planned"] = True
    else:
        normalized["is_planned"] = planned_series.astype(bool).reset_index(drop=True)

    _validate_absence_dates(normalized)

    normalized = normalized.drop_duplicates(
        subset=["employee_id", "date_from", "date_to", "type", "is_planned"],
        keep="first",
    ).reset_index(drop=True)

    return normalized


def explode_absences_by_day(
    abs_df: pd.DataFrame,
    min_date: "datetime.date | None" = None,
    max_date: "datetime.date | None" = None,
    absence_hours_h: float = 6.0,
) -> pd.DataFrame:
    """Explode absences into daily records within the provided horizon."""

    if absence_hours_h <= 0:
        raise ValueError("absence_hours_h deve essere positivo")

    if min_date is not None and max_date is not None and min_date > max_date:
        raise ValueError("min_date non può essere successivo a max_date")

    if abs_df.empty:
        return pd.DataFrame(
            columns=[
                "employee_id",
                "date",
                "type",
                "is_absent",
                "absence_hours_h",
                "is_planned",
            ]
        )

    absences = abs_df.copy()

    if min_date is not None:
        absences["date_from"] = absences["date_from"].apply(lambda d: max(d, min_date))
    if max_date is not None:
        absences["date_to"] = absences["date_to"].apply(lambda d: min(d, max_date))

    absences = absences[absences["date_from"] <= absences["date_to"]].copy()
    if absences.empty:
        return pd.DataFrame(
            columns=[
                "employee_id",
                "date",
                "type",
                "is_absent",
                "absence_hours_h",
                "is_planned",
            ]
        )

    planned_series = absences.get("is_planned")
    records = []
    for row in absences.itertuples(index=False):
        day_range = pd.date_range(row.date_from, row.date_to, freq="D")
        for day in day_range:
            is_planned = True
            if planned_series is not None:
                is_planned = bool(getattr(row, "is_planned"))
            records.append(
                {
                    "employee_id": row.employee_id,
                    "date": day.date(),
                    "type": row.type,
                    "is_absent": True,
                    "absence_hours_h": float(absence_hours_h),
                    "is_planned": is_planned,
                }
            )

    exploded = pd.DataFrame.from_records(records)
    exploded = exploded.drop_duplicates(subset=["employee_id", "date"], keep="last")
    exploded = exploded.sort_values(["employee_id", "date"]).reset_index(drop=True)

    return exploded


def get_absence_hours_from_config(config: dict) -> float:
    """Return the configured absence hours with validation."""

    payroll_cfg = config.get("payroll")
    if payroll_cfg is None:
        payroll_cfg = {}
    if not isinstance(payroll_cfg, dict):
        raise ValueError("config['payroll'] deve essere un dizionario valido")

    if "absence_hours_h" in payroll_cfg:
        logger.warning(
            "config: payroll.absence_hours_h è deprecato e verrà rimosso in futuro; "
            "migrare alla nuova configurazione delle assenze"
        )

    raw_value = payroll_cfg.get("absence_hours_h", 6.0)

    try:
        absence_hours = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Valore non numerico per payroll.absence_hours_h") from exc

    if absence_hours <= 0:
        raise ValueError("Le ore di assenza devono essere un numero positivo")

    return absence_hours
