from __future__ import annotations

import pandas as pd


_ABSENCE_REQUIRED_COLUMNS = {
    "employee_id",
    "date_from",
    "date_to",
    "type",
}


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
    _validate_absence_dates(normalized)

    normalized = normalized.drop_duplicates(
        subset=["employee_id", "date_from", "date_to", "type"], keep="first"
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
            columns=["employee_id", "date", "type", "is_absent", "absence_hours_h"]
        )

    absences = abs_df.copy()

    if min_date is not None:
        absences["date_from"] = absences["date_from"].apply(lambda d: max(d, min_date))
    if max_date is not None:
        absences["date_to"] = absences["date_to"].apply(lambda d: min(d, max_date))

    absences = absences[absences["date_from"] <= absences["date_to"]].copy()
    if absences.empty:
        return pd.DataFrame(
            columns=["employee_id", "date", "type", "is_absent", "absence_hours_h"]
        )

    records = []
    for row in absences.itertuples(index=False):
        day_range = pd.date_range(row.date_from, row.date_to, freq="D")
        for day in day_range:
            records.append(
                {
                    "employee_id": row.employee_id,
                    "date": day.date(),
                    "type": row.type,
                    "is_absent": True,
                    "absence_hours_h": float(absence_hours_h),
                }
            )

    exploded = pd.DataFrame.from_records(records)
    exploded = exploded.drop_duplicates(subset=["employee_id", "date"], keep="last")
    exploded = exploded.sort_values(["employee_id", "date"]).reset_index(drop=True)

    return exploded


def build_absence_masks(
    shift_slots: pd.DataFrame,
    abs_by_day: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build assignment and night-slot ban masks from absence information."""

    if shift_slots.empty or abs_by_day.empty:
        empty_assign = pd.DataFrame(columns=["employee_id", "date"])
        empty_nights = pd.DataFrame(columns=["employee_id", "slot_id"])
        return empty_assign, empty_nights

    working_abs = abs_by_day.loc[:, ["employee_id", "date"]].copy()
    working_abs["date"] = pd.to_datetime(working_abs["date"]).dt.date

    def _strip_timezone(series: pd.Series) -> pd.Series:
        tz = getattr(series.dt, "tz", None)
        if tz is None:
            return series
        return series.dt.tz_convert(None)

    start_dates = shift_slots[["slot_id", "employee_id", "start_dt"]].copy()
    start_dates["date"] = _strip_timezone(start_dates["start_dt"]).dt.date
    start_dates = start_dates.drop(columns=["start_dt"])

    df_absent_on_date = (
        start_dates.merge(
            working_abs,
            on=["employee_id", "date"],
            how="inner",
            validate="many_to_many",
        )[["employee_id", "date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    end_dates = shift_slots[["slot_id", "employee_id", "end_dt", "is_night"]].copy()
    end_dates["date"] = _strip_timezone(end_dates["end_dt"]).dt.date

    merged_nights = end_dates.merge(
        working_abs,
        on=["employee_id", "date"],
        how="inner",
        validate="many_to_many",
    )
    df_banned_night_slots = (
        merged_nights.loc[merged_nights["is_night"].astype(bool), ["employee_id", "slot_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_absent_on_date, df_banned_night_slots


def get_absence_hours_from_config(config: dict) -> float:
    """Return the configured absence hours with validation."""

    payroll_cfg = config.get("payroll")
    if payroll_cfg is None:
        payroll_cfg = {}
    if not isinstance(payroll_cfg, dict):
        raise ValueError("config['payroll'] deve essere un dizionario valido")

    raw_value = payroll_cfg.get("absence_hours_h", 6.0)

    try:
        absence_hours = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Valore non numerico per payroll.absence_hours_h") from exc

    if absence_hours <= 0:
        raise ValueError("Le ore di assenza devono essere un numero positivo")

    return absence_hours
