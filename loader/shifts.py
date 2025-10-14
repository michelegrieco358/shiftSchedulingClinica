from __future__ import annotations

import os
import re
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd

from .utils import (
    LoaderError,
    TURNI_DOMANDA,
    _ensure_cols,
    _resolve_allowed_departments,
    _resolve_allowed_roles,
)


def load_shifts(path: str) -> pd.DataFrame:
    """Carica e valida turni dal CSV con controlli su orari e durate."""
    hhmm = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"shift_id", "start", "end", "break_min", "duration_min", "crosses_midnight"},
        "shifts.csv",
    )

    for c in ["shift_id", "start", "end"]:
        df[c] = df[c].astype(str).str.strip()

    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="raise").astype(int)
    df["break_min"] = pd.to_numeric(df["break_min"], errors="raise").astype(int)
    df["crosses_midnight"] = pd.to_numeric(df["crosses_midnight"], errors="raise").astype(int)

    if (df["break_min"] < 0).any():
        bad = df.loc[df["break_min"] < 0, "shift_id"].unique().tolist()
        raise LoaderError(
            f"shifts.csv: break_min non può essere negativo per i turni: {bad}"
        )

    bad_cm = sorted(set(df["crosses_midnight"].unique()) - {0, 1})
    if bad_cm:
        raise LoaderError(
            f"shifts.csv: crosses_midnight deve essere 0 o 1, trovati: {bad_cm}"
        )

    key_cols = [
        "shift_id",
        "start",
        "end",
        "break_min",
        "crosses_midnight",
    ]
    if df["shift_id"].duplicated().any():
        grp = df.groupby("shift_id")[key_cols].nunique()
        diverging = grp[(grp > 1).any(axis=1)]
        if not diverging.empty:
            raise LoaderError(
                "shifts.csv: shift_id duplicati con definizioni diverse: "
                + ", ".join(diverging.index.tolist())
            )
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    zero_duration_shifts = {"R", "SN", "F"}
    mask_zero = df["shift_id"].isin(zero_duration_shifts)
    if (
        not df.loc[mask_zero, "duration_min"].eq(0).all()
        or not df.loc[mask_zero, "break_min"].eq(0).all()
        or not df.loc[mask_zero, "crosses_midnight"].eq(0).all()
    ):
        raise LoaderError(
            "shifts.csv: R, SN e F devono avere duration_min=0 e crosses_midnight=0"
        )

    if (df.loc[mask_zero, ["start", "end"]] != "").any().any():
        raise LoaderError("shifts.csv: R, SN e F devono avere start/end vuoti")

    mask_nzero = ~mask_zero
    if (df.loc[mask_nzero, "duration_min"] <= 0).any():
        bad = (
            df.loc[mask_nzero & (df["duration_min"] <= 0), "shift_id"].unique().tolist()
        )
        raise LoaderError(
            f"shifts.csv: turni con duration_min <= 0 non ammessi (eccetto R/SN): {bad}"
        )

    bad_start = df.loc[mask_nzero, "start"].apply(
        lambda s: bool(hhmm.fullmatch(s))
    ).eq(False)
    bad_end = df.loc[mask_nzero, "end"].apply(
        lambda s: bool(hhmm.fullmatch(s))
    ).eq(False)
    if bad_start.any() or bad_end.any():
        bad_rows = df.loc[mask_nzero & (bad_start | bad_end), ["shift_id", "start", "end"]]
        raise LoaderError(
            f"shifts.csv: start/end non validi (HH:MM) per turni non-zero:\n{bad_rows}"
        )

    def to_minutes(s: str) -> int:
        """Converte stringa HH:MM in minuti totali."""
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    computed_duration: dict[int, int] = {}
    for row in df.loc[
        mask_nzero, ["shift_id", "start", "end", "crosses_midnight", "break_min"]
    ].itertuples():
        idx = row.Index
        sid = row.shift_id
        s = row.start
        e = row.end
        cm = int(row.crosses_midnight)
        pause = int(row.break_min)
        sm = to_minutes(s)
        em = to_minutes(e)
        if cm == 0 and not (em > sm):
            raise LoaderError(
                f"shifts.csv: per turno {sid} crosses_midnight=0 ma end <= start ({e} <= {s})"
            )
        if cm == 1 and not (em < sm):
            raise LoaderError(
                f"shifts.csv: per turno {sid} crosses_midnight=1 ma end >= start ({e} >= {s})"
            )
        raw_duration = em - sm if cm == 0 else (24 * 60 - sm + em)
        if pause >= raw_duration:
            raise LoaderError(
                "shifts.csv: break_min %s per turno %s non può essere >= durata effettiva %s"
                % (pause, sid, raw_duration)
            )
        computed_duration[idx] = raw_duration - pause

    def to_timedelta_or_nat(s: str) -> pd.Timedelta | pd.NaTType:
        """Converte stringa HH:MM in timedelta o NaT se vuota."""
        if not s:
            return pd.NaT
        h, m = s.split(":")
        return pd.to_timedelta(int(h), unit="h") + pd.to_timedelta(int(m), unit="m")

    df["start_time"] = df["start"].apply(to_timedelta_or_nat)
    df["end_time"] = df["end"].apply(to_timedelta_or_nat)

    for idx, duration in computed_duration.items():
        df.at[idx, "duration_min"] = duration

    return df[[
        "shift_id",
        "start",
        "end",
        "break_min",
        "duration_min",
        "crosses_midnight",
        "start_time",
        "end_time",
    ]]


def _parse_hhmm_optional(value: str, label: str) -> pd.Timedelta | pd.NaTType:
    """Converte stringa HH:MM opzionale in Timedelta o NaT."""
    s = str(value).strip()
    if not s:
        return pd.NaT
    if not re.fullmatch(r"^(?:[01]\d|2[0-3]):[0-5]\d$", s):
        raise LoaderError(f"{label}: orario non valido (HH:MM): {value!r}")
    h, m = s.split(":")
    return pd.to_timedelta(int(h), unit="h") + pd.to_timedelta(int(m), unit="m")


def _coerce_optional_bool(value: Any, label: str) -> bool | None:
    """Converte valore generico in bool accettando vuoti."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "":
            return None
        if s in {"true", "1", "yes", "y", "si", "s"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    elif isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(int(value))
    elif isinstance(value, bool):
        return value
    raise LoaderError(f"{label}: valore booleano non riconosciuto ({value!r})")


def load_department_shift_map(
    path: str, defaults: dict[str, Any], shifts_df: pd.DataFrame
) -> pd.DataFrame:
    """Carica la mappa dei turni di reparto con eventuali override."""

    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "reparto_id",
                "shift_code",
                "enabled",
                "start_override",
                "end_override",
                "start_override_time",
                "end_override_time",
            ]
        )

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(
        df,
        {"reparto_id", "shift_code", "enabled", "start_override", "end_override"},
        "reparto_shift_map.csv",
    )

    for col in ["reparto_id", "shift_code", "enabled", "start_override", "end_override"]:
        df[col] = df[col].astype(str).str.strip()

    allowed_departments = set(_resolve_allowed_departments(defaults))
    bad_depts = sorted(set(df["reparto_id"].unique()) - allowed_departments)
    if bad_depts:
        raise LoaderError(
            "reparto_shift_map.csv: reparti non ammessi rispetto alla config: "
            f"{bad_depts}"
        )

    known_shifts = set(shifts_df["shift_id"].unique())
    bad_shifts = sorted(set(df["shift_code"].unique()) - known_shifts)
    if bad_shifts:
        raise LoaderError(
            "reparto_shift_map.csv: shift_code non presente nel catalogo globale (shifts.csv): "
            f"{bad_shifts}"
        )

    enabled: list[bool] = []
    start_td: list[pd.Timedelta | pd.NaTType] = []
    end_td: list[pd.Timedelta | pd.NaTType] = []
    for idx, row in df.iterrows():
        enabled_val = _coerce_optional_bool(row["enabled"], "reparto_shift_map.csv: enabled")
        enabled.append(True if enabled_val is None else bool(enabled_val))
        start_td.append(
            _parse_hhmm_optional(
                row["start_override"], "reparto_shift_map.csv: start_override"
            )
        )
        end_td.append(
            _parse_hhmm_optional(
                row["end_override"], "reparto_shift_map.csv: end_override"
            )
        )

    df["enabled"] = enabled
    df["start_override_time"] = start_td
    df["end_override_time"] = end_td

    if df.duplicated(subset=["reparto_id", "shift_code"]).any():
        dup = df[
            df.duplicated(subset=["reparto_id", "shift_code"], keep=False)
        ].sort_values(["reparto_id", "shift_code"])
        raise LoaderError(
            "reparto_shift_map.csv: duplicati non ammessi su (reparto_id, shift_code):\n"
            f"{dup[['reparto_id','shift_code']]}"
        )

    return df[
        [
            "reparto_id",
            "shift_code",
            "enabled",
            "start_override",
            "end_override",
            "start_override_time",
            "end_override_time",
        ]
    ]


def _compute_raw_duration_minutes(start: pd.Timedelta, end: pd.Timedelta) -> int:
    """Calcola la durata in minuti tra start ed end, gestendo il cross-midnight."""
    base_day = pd.to_timedelta(1, unit="D")
    delta = end - start
    if delta <= pd.to_timedelta(0):
        delta = base_day - start + end
    minutes = int(delta.total_seconds() // 60)
    if minutes <= 0:
        raise LoaderError(
            "Durata turno non positiva dopo l'applicazione degli override (start=%s, end=%s)"
            % (start, end)
        )
    return minutes


def _compute_duration_minutes(
    start: pd.Timedelta, end: pd.Timedelta, break_min: int
) -> int:
    """Restituisce la durata effettiva sottraendo la pausa."""
    minutes = _compute_raw_duration_minutes(start, end)
    if break_min < 0:
        raise LoaderError(
            "Durata turno non valida: break_min negativo (%s minuti)" % break_min
        )
    if break_min >= minutes:
        raise LoaderError(
            "Durata turno non valida: break_min %s >= durata %s" % (break_min, minutes)
        )
    return minutes - break_min


def _is_night_shift(
    start: pd.Timedelta, end: pd.Timedelta, crosses_midnight: bool
) -> bool:
    """Determina se il turno va considerato notturno basandosi sugli orari."""
    if crosses_midnight:
        return True
    start_minutes = start.components.hours * 60 + start.components.minutes
    end_minutes = end.components.hours * 60 + end.components.minutes
    if start_minutes >= 22 * 60:
        return True
    if end_minutes <= 6 * 60:
        return True
    return False


def _local_day_start(calendar_date: pd.Timestamp, tz: ZoneInfo) -> pd.Timestamp:
    """Restituisce la mezzanotte locale (timezone-aware) del giorno indicato."""

    ts = pd.Timestamp(calendar_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts.normalize()


def build_shift_slots(
    month_plan_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    dept_map_df: pd.DataFrame,
    defaults: dict[str, Any],
) -> pd.DataFrame:
    """Costruisce gli slot di turno giornalieri applicando override di reparto."""

    if month_plan_df.empty:
        return pd.DataFrame(
            columns=[
                "slot_id",
                "data",
                "data_dt",
                "reparto_id",
                "shift_code",
                "coverage_code",
                "start_time",
                "end_time",
                "duration_min",
                "crosses_midnight",
                "is_night",
                "start_dt",
                "end_dt",
                "shift_label",
            ]
        )

    allowed_departments = set(_resolve_allowed_departments(defaults))
    shift_lookup = shifts_df.set_index("shift_id")
    dept_map = {
        (row.reparto_id, row.shift_code): row
        for row in dept_map_df.itertuples(index=False)
    }

    tz = ZoneInfo("Europe/Rome")
    rows: list[dict[str, Any]] = []

    for idx, mp_row in month_plan_df.iterrows():
        reparto_id = str(mp_row.get("reparto_id", "")).strip()
        if reparto_id == "":
            raise LoaderError(
                "month_plan.csv: la colonna reparto_id è obbligatoria per tutte le righe"
            )
        if reparto_id not in allowed_departments:
            raise LoaderError(
                "month_plan.csv: reparto_id '%s' non è fra quelli ammessi in config"
                % reparto_id
            )

        shift_code = str(mp_row.get("shift_code", "")).strip()
        if shift_code == "":
            raise LoaderError(
                "month_plan.csv: shift_code obbligatorio (riga indice %s)" % idx
            )
        if shift_code not in shift_lookup.index:
            raise LoaderError(
                "month_plan.csv: shift_code '%s' non definito nel catalogo globale"
                % shift_code
            )

        coverage_code = str(mp_row.get("coverage_code", "")).strip()
        if coverage_code == "":
            raise LoaderError(
                "month_plan.csv: coverage_code obbligatorio (riga indice %s)" % idx
            )

        calendar_date = mp_row.get("data_dt")
        if pd.isna(calendar_date):
            raise LoaderError(
                "month_plan.csv: colonna data_dt mancante — assicurarsi di aver agganciato il calendario"
            )

        base = shift_lookup.loc[shift_code]
        override = dept_map.get((reparto_id, shift_code))
        requires_department_opt_in = (
            base["duration_min"] > 0 and shift_code not in TURNI_DOMANDA
        )
        if override is None and requires_department_opt_in:
            raise LoaderError(
                "month_plan.csv: turno '%s' non abilitato per reparto '%s' (manca riga in reparto_shift_map.csv)"
                % (shift_code, reparto_id)
            )
        if override is not None and not bool(override.enabled):
            raise LoaderError(
                "month_plan.csv: turno '%s' disabilitato per reparto '%s' secondo reparto_shift_map"
                % (shift_code, reparto_id)
            )

        start_time = (
            override.start_override_time
            if override is not None and pd.notna(override.start_override_time)
            else base["start_time"]
        )
        end_time = (
            override.end_override_time
            if override is not None and pd.notna(override.end_override_time)
            else base["end_time"]
        )

        if pd.isna(start_time) or pd.isna(end_time):
            raise LoaderError(
                "month_plan.csv: turno '%s' per reparto '%s' non ha orari definiti (start/end). "
                "Definire gli orari in shifts.csv oppure specificare gli override."
                % (shift_code, reparto_id)
            )

        crosses_midnight = bool(end_time < start_time)
        break_min = int(base.get("break_min", 0))
        duration_min = _compute_duration_minutes(start_time, end_time, break_min)

        day_start = _local_day_start(calendar_date, tz)
        start_dt = day_start + start_time
        end_dt = day_start + (
            pd.to_timedelta(1, unit="D") if crosses_midnight else pd.Timedelta(0)
        ) + end_time

        rows.append(
            {
                "data": mp_row.get("data"),
                "data_dt": pd.Timestamp(calendar_date),
                "reparto_id": reparto_id,
                "shift_code": shift_code,
                "coverage_code": coverage_code,
                "start_time": start_time,
                "end_time": end_time,
                "duration_min": duration_min,
                "crosses_midnight": crosses_midnight,
                "is_night": _is_night_shift(start_time, end_time, crosses_midnight),
                "start_dt": pd.Timestamp(start_dt),
                "end_dt": pd.Timestamp(end_dt),
                "shift_label": f"{reparto_id}_{shift_code}",
            }
        )

    slots_df = pd.DataFrame(rows)
    if slots_df.empty:
        return slots_df

    slots_df = slots_df.sort_values(
        ["data_dt", "reparto_id", "shift_code", "coverage_code"]
    ).reset_index(drop=True)
    slots_df.insert(0, "slot_id", range(1, len(slots_df) + 1))
    return slots_df


def load_shift_role_eligibility(
    path: str, employees_df: pd.DataFrame, shifts_df: pd.DataFrame, defaults: dict
) -> pd.DataFrame:
    """Carica e valida l'idoneità dei ruoli per ogni turno."""
    df = pd.read_csv(path, dtype=str).fillna("")

    # Supportiamo sia lo schema legacy (shift_id/ruolo) sia quello esteso
    # (shift_code/role/allowed).
    has_shift_code = "shift_code" in df.columns
    has_shift_id = "shift_id" in df.columns
    has_role = "role" in df.columns
    has_ruolo = "ruolo" in df.columns

    if not (has_shift_code or has_shift_id):
        raise LoaderError(
            "shift_role_eligibility.csv deve contenere la colonna 'shift_code' o 'shift_id'"
        )
    if not (has_role or has_ruolo):
        raise LoaderError(
            "shift_role_eligibility.csv deve contenere la colonna 'role' o 'ruolo'"
        )

    columns_required = set()
    if has_shift_code:
        columns_required.add("shift_code")
    if has_shift_id:
        columns_required.add("shift_id")
    if has_role:
        columns_required.add("role")
    if has_ruolo:
        columns_required.add("ruolo")
    _ensure_cols(df, columns_required, "shift_role_eligibility.csv")

    if has_shift_id and not has_shift_code:
        df = df.rename(columns={"shift_id": "shift_code"})
    if has_ruolo and not has_role:
        df = df.rename(columns={"ruolo": "role"})

    df["shift_code"] = df["shift_code"].astype(str).str.strip().str.upper()
    df["role"] = df["role"].astype(str).str.strip().str.upper()

    if (df["shift_code"] == "").any():
        bad = df.index[df["shift_code"] == ""].tolist()[:5]
        raise LoaderError(
            "shift_role_eligibility.csv: shift_code vuoti nelle righe (prime occorrenze): "
            f"{bad}"
        )
    if (df["role"] == "").any():
        bad = df.index[df["role"] == ""].tolist()[:5]
        raise LoaderError(
            "shift_role_eligibility.csv: role vuoti nelle righe (prime occorrenze): "
            f"{bad}"
        )

    # Determiniamo e normalizziamo la colonna allowed.
    def _parse_allowed(value: str) -> bool:
        s = str(value).strip().lower()
        if s in {"", "nan"}:
            return True
        if s in {"true", "1", "yes", "y", "si", "s"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        raise LoaderError(
            f"shift_role_eligibility.csv: valore booleano non riconosciuto in 'allowed': {value!r}"
        )

    if "allowed" in df.columns:
        df["allowed"] = df["allowed"].apply(_parse_allowed)
    else:
        df["allowed"] = True

    # Usa config come fonte di verità per i ruoli ammessi
    allowed_roles = _resolve_allowed_roles(
        defaults, fallback_roles=employees_df["role"].unique()
    )
    known_roles = {str(role).strip().upper() for role in allowed_roles}
    bad_roles = sorted(set(df["role"].unique()) - known_roles)
    if bad_roles:
        raise LoaderError(
            "shift_role_eligibility.csv: ruoli non ammessi rispetto alla config: "
            f"{bad_roles}"
        )

    known_shifts = {
        str(shift).strip().upper() for shift in shifts_df["shift_id"].astype(str).tolist()
    }
    bad_shifts = sorted(set(df["shift_code"].unique()) - known_shifts)
    if bad_shifts:
        raise LoaderError(
            "shift_role_eligibility.csv: shift_code sconosciuti rispetto a shifts.csv: "
            f"{bad_shifts}"
        )

    df = df.drop_duplicates(subset=["shift_code", "role"], keep="last").reset_index(drop=True)

    demand_shifts_in_catalog = sorted(TURNI_DOMANDA & known_shifts)
    for sid in demand_shifts_in_catalog:
        if df.loc[(df["shift_code"] == sid) & df["allowed"].fillna(False)].empty:
            raise LoaderError(
                "shift_role_eligibility.csv: nessun ruolo idoneo definito per turno di domanda "
                f"'{sid}'"
            )

    df = df.sort_values(["shift_code", "role"]).reset_index(drop=True)
    return df[["shift_code", "role", "allowed"]]
