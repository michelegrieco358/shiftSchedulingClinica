
"""
Loader clinica — 
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, Set
import pandas as pd
from datetime import datetime, date, timedelta
import yaml

TURNI_DOMANDA: Set[str] = {"M", "P", "N"}

class LoaderError(Exception):
    pass

@dataclass
class LoadedData:
    cfg: dict
    calendar_df: pd.DataFrame
    employees_df: pd.DataFrame
    shifts_df: pd.DataFrame
    eligibility_df: pd.DataFrame
    month_plan_df: pd.DataFrame
    groups_total_expanded: pd.DataFrame
    groups_role_min_expanded: pd.DataFrame
    history_df: pd.DataFrame
    availability_df: pd.DataFrame
    leaves_df: pd.DataFrame
    leaves_days_df: pd.DataFrame
    holidays_df: pd.DataFrame
    role_dept_pools_df: pd.DataFrame
    dept_compat_df: pd.DataFrame

def _parse_date(s: str) -> date:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()

def _ensure_cols(df: pd.DataFrame, required: Set[str], label: str):
    missing = required - set(df.columns)
    if missing:
        raise LoaderError(f"{label}: colonne mancanti {sorted(missing)}")


def _resolve_allowed_roles(defaults: dict, fallback_roles: Iterable[str] | None = None) -> list[str]:
    allowed_roles_cfg = defaults.get("allowed_roles", None)
    roles: list[str]
    if isinstance(allowed_roles_cfg, str):
        roles = [
            x.strip()
            for x in allowed_roles_cfg.replace(",", "|").split("|")
            if x.strip()
        ]
    elif isinstance(allowed_roles_cfg, (list, tuple, set)):
        roles = [str(x).strip() for x in allowed_roles_cfg if str(x).strip()]
    else:
        roles = []

    if not roles and fallback_roles is not None:
        roles = [str(x).strip() for x in fallback_roles if str(x).strip()]

    # Dedup preservando ordine
    seen = set()
    deduped = []
    for role in roles:
        if role not in seen:
            seen.add(role)
            deduped.append(role)
    return deduped


def _resolve_allowed_departments(defaults: dict) -> list[str]:
    departments_cfg = defaults.get("departments", None)
    if isinstance(departments_cfg, str):
        departments = [
            x.strip()
            for x in departments_cfg.replace(",", "|").split("|")
            if x.strip()
        ]
    elif isinstance(departments_cfg, (list, tuple, set)):
        departments = [str(x).strip() for x in departments_cfg if str(x).strip()]
    else:
        departments = []

    if not departments:
        raise LoaderError(
            "config: defaults.departments deve essere una lista non vuota di reparti ammessi"
        )

    seen = set()
    deduped = []
    for dept in departments:
        if dept not in seen:
            seen.add(dept)
            deduped.append(dept)
    return deduped

def build_calendar(start_date: date, end_date: date, holidays_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Costruisce un calendario esteso per la pianificazione dei turni clinici.
    
    Crea un DataFrame con tutti i giorni dall'inizio della settimana precedente
    al periodo di pianificazione fino alla data di fine.
    
    Args:
        start_date (date): Data di inizio del periodo di pianificazione
        end_date (date): Data di fine del periodo di pianificazione
        
    Returns:
        pd.DataFrame: DataFrame con le seguenti colonne:
            - data: Data in formato ISO (YYYY-MM-DD)
            - data_dt: Data in formato datetime64[ns]
            - dow_iso: Giorno della settimana ISO (1=lunedì, 7=domenica)
            - week_start_date: Data di inizio settimana (lunedì) in formato ISO
            - week_start_date_dt: Data di inizio settimana come datetime64[ns]
            - week_id: Identificativo della settimana (uguale a week_start_date)
            - is_in_horizon: Boolean che indica se la data è nel periodo di pianificazione
            - week_idx: Indice numerico progressivo della settimana (0, 1, 2, ...)
            - cal_start: Data di inizio del calendario esteso (ISO)
            - cal_start_dt: Data di inizio calendario come datetime64[ns]
            - is_weekend: Boolean che indica se la data è sabato/domenica
            - is_weekday_holiday: Boolean per i festivi infrasettimanali da holidays.csv
            - holiday_desc: Descrizione del festivo infrasettimanale (stringa vuota se non festivo)
            
    Note:
        Il calendario inizia dal lunedì della settimana che contiene start_date
        o da 6 giorni prima di start_date (il minimo tra i due), per garantire
        una visibilità completa per la pianificazione dei turni.
    """
    
    prev_week_start = start_date - timedelta(days=(start_date.isoweekday() - 1))
    six_days_before = start_date - timedelta(days=6)
    cal_start = min(prev_week_start, six_days_before)
    rows = []
    d = cal_start
    while d <= end_date:
        dow_iso = d.isoweekday()
        week_start_date = d - timedelta(days=dow_iso - 1)
        rows.append({
            "data": d.isoformat(),
            "dow_iso": dow_iso,
            "week_start_date": week_start_date.isoformat(),
            "week_id": week_start_date.isoformat(),
            "is_in_horizon": (start_date <= d <= end_date),
        })
        d += timedelta(days=1)
    cal = pd.DataFrame(rows)
    wk_map = {ws: i for i, ws in enumerate(sorted(cal["week_start_date"].unique()))}
    cal["week_idx"] = cal["week_start_date"].map(wk_map)
    cal["cal_start"] = cal_start.isoformat()
    cal["data_dt"] = pd.to_datetime(cal["data"], format="%Y-%m-%d")
    cal["week_start_date_dt"] = pd.to_datetime(cal["week_start_date"], format="%Y-%m-%d")
    cal["cal_start_dt"] = pd.to_datetime(cal_start)
    cal["is_weekend"] = cal["dow_iso"].isin([6, 7])

    holiday_desc_col = pd.Series([""] * len(cal), index=cal.index)
    if holidays_df is not None and not holidays_df.empty:
        holidays_unique = holidays_df.drop_duplicates(subset=["data"], keep="first")
        cal = cal.merge(
            holidays_unique[["data", "descrizione"]].rename(columns={"descrizione": "holiday_desc"}),
            on="data",
            how="left",
        )
        holiday_desc_col = cal["holiday_desc"].fillna("")
    cal["holiday_desc"] = holiday_desc_col.astype(str)
    cal["is_weekday_holiday"] = cal["holiday_desc"].str.strip().ne("")
    return cal

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    try:
        _ = _parse_date(cfg["horizon"]["start_date"])
        _ = _parse_date(cfg["horizon"]["end_date"])
    except Exception as e:
        raise LoaderError(f"config: horizon/start_date,end_date mancanti o non validi: {e}")
    return cfg

def load_holidays(path: str) -> pd.DataFrame:
    """Carica holidays.csv se presente e restituisce [data, data_dt, descrizione]."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["data", "data_dt", "descrizione"])

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data", "descrizione"}, "holidays.csv")

    df["data"] = df["data"].astype(str).str.strip()
    df["descrizione"] = df["descrizione"].astype(str).str.strip()

    if (df["data"] == "").any():
        raise LoaderError("holidays.csv: la colonna 'data' non può contenere valori vuoti")

    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"holidays.csv: formato data non valido: {exc}")

    dup_dates = df[df["data"].duplicated(keep=False)]["data"].unique()
    if len(dup_dates) > 0:
        raise LoaderError(
            "holidays.csv: date duplicate non ammesse: " + ", ".join(sorted(dup_dates))
        )

    return df[["data", "data_dt", "descrizione"]].sort_values("data").reset_index(drop=True)

def load_employees(path: str, defaults: dict) -> pd.DataFrame:
    """
    Legge employees.csv e applica:
    - unicità employee_id
    - validazione ruoli rispetto a defaults['allowed_roles'] (se presente)
    - ore dovute: se vuote, usa defaults['contract_hours_by_role_h'][ruolo]
    - divieto di valori negativi (eccetto saldo iniziale che può essere < 0)
    - conversione ore → minuti per i campi rilevanti
    - arricchimento con contatori equità weekend/festivi (default 0 se mancanti)
    """
    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(
        df,
        {
            "employee_id",
            "nome",
            "ruolo",
            "reparto",
            "ore_dovute_mese_h",
            "saldo_prog_iniziale_h",
        },
        "employees.csv",
    )

    # --- Unicità employee_id ---
    if df["employee_id"].duplicated().any():
        dups = df[df["employee_id"].duplicated(keep=False)].sort_values("employee_id")
        raise LoaderError(
            f"employees.csv: employee_id duplicati:\n{dups[['employee_id','nome','ruolo']]}"
        )

    # --- Ruoli ammessi da config (opzionale ma consigliato) ---
    # defaults['allowed_roles'] può essere lista oppure stringa pipe/comma-separated.
    allowed_roles = _resolve_allowed_roles(defaults, fallback_roles=df["ruolo"].unique())

    bad_roles = sorted(set(df["ruolo"].unique()) - set(allowed_roles))
    if bad_roles:
        raise LoaderError(f"employees.csv: ruoli non ammessi rispetto alla config: {bad_roles}")

    # --- Reparti ammessi e obbligatorietà del campo ---
    allowed_departments = _resolve_allowed_departments(defaults)
    df["reparto"] = df["reparto"].astype(str).str.strip()
    if (df["reparto"] == "").any():
        bad = df.loc[df["reparto"] == "", ["employee_id", "nome", "ruolo"]]
        raise LoaderError(
            "employees.csv: la colonna 'reparto' è obbligatoria e non può essere vuota. "
            f"Righe interessate:\n{bad}"
        )

    bad_departments = sorted(set(df["reparto"].unique()) - set(allowed_departments))
    if bad_departments:
        raise LoaderError(
            "employees.csv: reparti non ammessi rispetto alla config (defaults.departments): "
            f"{bad_departments}"
        )

    # --- Helper di parsing ---
    def parse_hours_nonneg(x, field_name: str) -> float:
        """Converte ore (stringa/numero) in float >=0. Errore se non numerico o negativo."""
        s = str(x).strip()
        try:
            v = float(s)
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{field_name}': {x!r}")
        if v < 0:
            raise LoaderError(f"employees.csv: valore negativo non ammesso in '{field_name}': {v}")
        return v

    def to_min_from_hours(v_hours: float) -> int:
        return int(round(v_hours * 60.0))

    def parse_hours_allow_negative(x, field_name: str) -> float:
        """Per saldo iniziale: consente negativo."""
        s = str(x).strip()
        try:
            return float(s)
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{field_name}': {x!r}")

    # --- Ore dovute: se vuote, prendi default per ruolo da config ---
    # defaults['contract_hours_by_role_h'] deve essere un dict {ruolo: ore_float}
    contract_by_role = defaults.get("contract_hours_by_role_h", {}) or {}

    dovuto_min = []
    for _, r in df.iterrows():
        raw = str(r["ore_dovute_mese_h"]).strip()
        ruolo = str(r["ruolo"]).strip()
        if raw != "":
            # usa il valore del CSV, validando non-negativo
            hours = parse_hours_nonneg(raw, "ore_dovute_mese_h")
        else:
            # usa il default per ruolo
            if ruolo not in contract_by_role:
                raise LoaderError(
                    f"employees.csv: ore_dovute_mese_h mancante e nessun default in config per ruolo '{ruolo}' "
                    "(defaults.contract_hours_by_role_h)"
                )
            hours = parse_hours_nonneg(contract_by_role[ruolo], f"contract_hours_by_role_h[{ruolo}]")
        dovuto_min.append(to_min_from_hours(hours))
    df["dovuto_min"] = dovuto_min

    # --- Saldo progressivo iniziale (può essere negativo) ---
    df["saldo_init_min"] = df["saldo_prog_iniziale_h"].apply(
        lambda x: to_min_from_hours(parse_hours_allow_negative(x, "saldo_prog_iniziale_h"))
    )

    # --- Colonne opzionali con default da config + validazione non-negativa ---
    # Se non presenti o vuote → usa default. Non possono essere negative.
    def get_hours_with_default(col_name: str, default_val) -> pd.Series:
        if col_name in df.columns:
            ser = df[col_name].astype(str).str.strip()
            ser = ser.where(ser != "", other=str(default_val))
        else:
            ser = pd.Series([str(default_val)] * len(df))
        # valida e converte in minuti
        hours = ser.apply(lambda x: parse_hours_nonneg(x, col_name))
        return hours.apply(to_min_from_hours)

    df["max_week_min"] = get_hours_with_default(
        "max_week_hours_h", defaults.get("max_week_hours_h", 60)
    )
    df["max_month_extra_min"] = get_hours_with_default(
        "max_month_extra_h", defaults.get("max_month_extra_h", 40)
    )

    # Notti: interi non negativi con default
    def get_int_with_default(col_name: str, default_val: int) -> pd.Series:
        if col_name in df.columns:
            ser = df[col_name].astype(str).str.strip()
            ser = ser.where(ser != "", other=str(default_val))
        else:
            ser = pd.Series([str(default_val)] * len(df))
        try:
            vals = ser.astype(float)  # accetta "3.0"
        except ValueError:
            raise LoaderError(f"employees.csv: valore non numerico in '{col_name}'")
        if (vals < 0).any():
            raise LoaderError(f"employees.csv: valore negativo non ammesso in '{col_name}'")
        # arrotonda e cast a int
        return vals.round().astype(int)

    df["max_nights_week"] = get_int_with_default(
        "max_nights_week", int(defaults.get("max_nights_week", 3))
    )
    df["max_nights_month"] = get_int_with_default(
        "max_nights_month", int(defaults.get("max_nights_month", 8))
    )

    # Contatori di equità (weekend/festivi): interi >= 0, default 0
    df["saturday_count_ytd"] = get_int_with_default("saturday_count_ytd", 0)
    df["sunday_count_ytd"] = get_int_with_default("sunday_count_ytd", 0)
    df["holiday_count_ytd"] = get_int_with_default("holiday_count_ytd", 0)

    # --- Ritorna solo le colonne utili ---
    return df[
        [
            "employee_id",
            "nome",
            "ruolo",
            "reparto",
            "dovuto_min",
            "saldo_init_min",
            "max_week_min",
            "max_month_extra_min",
            "max_nights_week",
            "max_nights_month",
            "saturday_count_ytd",
            "sunday_count_ytd",
            "holiday_count_ytd",
        ]
    ]


def load_shifts(path: str) -> pd.DataFrame:
    """
    Carica e valida il catalogo turni (shifts.csv).
    Requisiti:
      - colonne obbligatorie: shift_id, start, end, duration_min, crosses_midnight
      - shift_id univoci (duplicati identici vengono deduplicati; se divergenti -> errore)
      - R, SN e F: duration_min=0, crosses_midnight=0, start/end vuoti
      - turni con duration_min>0: start/end obbligatori in formato HH:MM
      - crosses_midnight ∈ {0,1}
      - coerenza base:
          * se crosses_midnight=0 -> end > start (stesso giorno)
          * se crosses_midnight=1 -> end < start (passa la mezzanotte)
    """
    import re
    hhmm = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"shift_id", "start", "end", "duration_min", "crosses_midnight"}, "shifts.csv")

    # Normalizza spazi
    for c in ["shift_id", "start", "end"]:
        df[c] = df[c].astype(str).str.strip()

    # Tipi numerici + validità di base
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="raise").astype(int)
    df["crosses_midnight"] = pd.to_numeric(df["crosses_midnight"], errors="raise").astype(int)

    # crosses_midnight deve essere 0/1
    bad_cm = sorted(set(df["crosses_midnight"].unique()) - {0, 1})
    if bad_cm:
        raise LoaderError(f"shifts.csv: crosses_midnight deve essere 0 o 1, trovati: {bad_cm}")

    # Gestione duplicati di shift_id:
    # - se righe con lo stesso shift_id differiscono per start/end/duration/crosses -> errore
    # - se identiche -> deduplica
    key_cols = ["shift_id", "start", "end", "duration_min", "crosses_midnight"]
    if df["shift_id"].duplicated().any():
        grp = df.groupby("shift_id")[key_cols].nunique()
        diverging = grp[(grp > 1).any(axis=1)]
        if not diverging.empty:
            raise LoaderError(
                "shifts.csv: shift_id duplicati con definizioni diverse: "
                + ", ".join(diverging.index.tolist())
            )
        # tutte le definizioni identiche -> teniamo una sola riga
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    # Regole speciali per turni a durata nulla (riposo/ferie/smonto notte)
    zero_duration_shifts = {"R", "SN", "F"}
    mask_zero = df["shift_id"].isin(zero_duration_shifts)
    if not df.loc[mask_zero, "duration_min"].eq(0).all() or not df.loc[mask_zero, "crosses_midnight"].eq(0).all():
        raise LoaderError(
            "shifts.csv: R, SN e F devono avere duration_min=0 e crosses_midnight=0"
        )

    # Per i turni a durata nulla richiediamo start/end vuoti (stringa vuota)
    if (df.loc[mask_zero, ["start", "end"]] != "").any().any():
        raise LoaderError("shifts.csv: R, SN e F devono avere start/end vuoti")

    # Per tutti gli altri shift_id: duration_min deve essere > 0
    mask_nzero = ~mask_zero
    if (df.loc[mask_nzero, "duration_min"] <= 0).any():
        bad = df.loc[mask_nzero & (df["duration_min"] <= 0), "shift_id"].unique().tolist()
        raise LoaderError(f"shifts.csv: turni con duration_min <= 0 non ammessi (eccetto R/SN): {bad}")

    # Turni non-zero: start/end obbligatori e in HH:MM
    bad_start = df.loc[mask_nzero, "start"].apply(lambda s: bool(hhmm.fullmatch(s))).eq(False)
    bad_end = df.loc[mask_nzero, "end"].apply(lambda s: bool(hhmm.fullmatch(s))).eq(False)
    if bad_start.any() or bad_end.any():
        bad_rows = df.loc[mask_nzero & (bad_start | bad_end), ["shift_id", "start", "end"]]
        raise LoaderError(f"shifts.csv: start/end non validi (HH:MM) per turni non-zero:\n{bad_rows}")

    # Coerenza con crosses_midnight rispetto agli orari
    def to_minutes(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    # Verifica: se cm=0, end > start; se cm=1, end < start
    for sid, s, e, cm in df.loc[mask_nzero, ["shift_id", "start", "end", "crosses_midnight"]].itertuples(index=False):
        sm = to_minutes(s)
        em = to_minutes(e)
        if cm == 0 and not (em > sm):
            raise LoaderError(f"shifts.csv: per turno {sid} crosses_midnight=0 ma end <= start ({e} <= {s})")
        if cm == 1 and not (em < sm):
            raise LoaderError(f"shifts.csv: per turno {sid} crosses_midnight=1 ma end >= start ({e} >= {s})")

    def to_timedelta_or_nat(s: str):
        if not s:
            return pd.NaT
        h, m = s.split(":")
        return pd.to_timedelta(int(h), unit="h") + pd.to_timedelta(int(m), unit="m")

    df["start_time"] = df["start"].apply(to_timedelta_or_nat)
    df["end_time"] = df["end"].apply(to_timedelta_or_nat)

    return df[[
        "shift_id",
        "start",
        "end",
        "duration_min",
        "crosses_midnight",
        "start_time",
        "end_time",
    ]]


def load_shift_role_eligibility(path: str, employees_df: pd.DataFrame, shifts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Carica e valida la mappa di idoneità (turno ↔ ruolo):
      - colonne obbligatorie: shift_id, ruolo
      - rimozione spazi, divieto di valori vuoti
      - ruoli devono esistere in employees_df
      - shift_id devono esistere in shifts_df
      - dedup su (shift_id, ruolo)
      - requisito: per ogni turno di domanda presente nel catalogo (M/P/N), almeno un ruolo idoneo
    Ritorna DataFrame con colonne ["shift_id","ruolo"], ordinato.
    """
    TURNI_DOMANDA = {"M", "P", "N"}

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"shift_id", "ruolo"}, "shift_role_eligibility.csv")

    # normalizza stringhe
    df["shift_id"] = df["shift_id"].astype(str).str.strip()
    df["ruolo"] = df["ruolo"].astype(str).str.strip()

    # nessun valore vuoto
    if (df["shift_id"] == "").any():
        bad = df.loc[df["shift_id"] == "", :].index.tolist()[:5]
        raise LoaderError(f"shift_role_eligibility.csv: shift_id vuoti nelle righe (prime occorrenze): {bad}")
    if (df["ruolo"] == "").any():
        bad = df.loc[df["ruolo"] == "", :].index.tolist()[:5]
        raise LoaderError(f"shift_role_eligibility.csv: ruolo vuoto nelle righe (prime occorrenze): {bad}")

    # ruoli noti (già validati in load_employees contro allowed_roles)
    known_roles = set(employees_df["ruolo"].unique())
    bad_roles = sorted(set(df["ruolo"].unique()) - known_roles)
    if bad_roles:
        raise LoaderError(f"shift_role_eligibility.csv: ruoli sconosciuti rispetto a employees.csv: {bad_roles}")

    # shift_id noti dal catalogo turni
    known_shifts = set(shifts_df["shift_id"].unique())
    bad_shifts = sorted(set(df["shift_id"].unique()) - known_shifts)
    if bad_shifts:
        raise LoaderError(f"shift_role_eligibility.csv: shift_id sconosciuti rispetto a shifts.csv: {bad_shifts}")

    # dedup coppie (shift_id, ruolo)
    before = len(df)
    df = df.drop_duplicates(subset=["shift_id", "ruolo"]).reset_index(drop=True)
    # (opzionale) si potrebbe loggare quanti duplicati sono stati rimossi:
    # removed = before - len(df)

    # requisito minimo: per ogni turno di domanda presente nel catalogo, almeno un ruolo idoneo
    demand_shifts_in_catalog = sorted(TURNI_DOMANDA & known_shifts)
    for sid in demand_shifts_in_catalog:
        if df.loc[df["shift_id"] == sid].empty:
            raise LoaderError(f"shift_role_eligibility.csv: nessun ruolo idoneo definito per turno di domanda '{sid}'")

    # ordina per leggibilità
    df = df.sort_values(["shift_id", "ruolo"]).reset_index(drop=True)
    return df[["shift_id", "ruolo"]]


def load_role_dept_pools(
    path: str, defaults: dict, employees_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Carica e valida la tabella di pool reparti ↔ ruolo.

    - colonne obbligatorie: ruolo, pool_id, reparto
    - ruolo deve essere ammesso da config (fallback ai ruoli presenti negli employees)
    - reparto deve appartenere al vocabolario defaults.departments
    - nessun valore vuoto e nessun duplicato su (ruolo, pool_id, reparto)
    """

    if not os.path.exists(path):
        return pd.DataFrame(columns=["ruolo", "pool_id", "reparto"])

    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"ruolo", "pool_id", "reparto"}, "role_dept_pools.csv")

    for col in ["ruolo", "pool_id", "reparto"]:
        df[col] = df[col].astype(str).str.strip()

    if (df[["ruolo", "pool_id", "reparto"]] == "").any().any():
        bad_rows = df.loc[(df[["ruolo", "pool_id", "reparto"]] == "").any(axis=1)]
        raise LoaderError(
            "role_dept_pools.csv: valori vuoti non ammessi nelle colonne ruolo/pool_id/reparto. "
            f"Righe interessate:\n{bad_rows}"
        )

    allowed_roles = set(
        _resolve_allowed_roles(defaults, fallback_roles=employees_df["ruolo"].unique())
    )
    allowed_departments = set(_resolve_allowed_departments(defaults))

    bad_roles = sorted(set(df["ruolo"].unique()) - allowed_roles)
    if bad_roles:
        raise LoaderError(
            "role_dept_pools.csv: ruoli non ammessi rispetto alla config: "
            f"{bad_roles}"
        )

    bad_departments = sorted(set(df["reparto"].unique()) - allowed_departments)
    if bad_departments:
        raise LoaderError(
            "role_dept_pools.csv: reparti non ammessi rispetto alla config: "
            f"{bad_departments}"
        )

    if df.duplicated(subset=["ruolo", "pool_id", "reparto"]).any():
        dup = df[
            df.duplicated(subset=["ruolo", "pool_id", "reparto"], keep=False)
        ].sort_values(["ruolo", "pool_id", "reparto"])
        raise LoaderError(
            "role_dept_pools.csv: duplicati non ammessi su (ruolo, pool_id, reparto):\n"
            f"{dup}"
        )

    return df.sort_values(["ruolo", "pool_id", "reparto"]).reset_index(drop=True)


def build_department_compatibility(
    defaults: dict, pools_df: pd.DataFrame, employees_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Costruisce la tabella delle compatibilità ruolo/reparto (home → target).

    Include sempre l'identità (home == target) per tutti i reparti ammessi,
    per ciascun ruolo ammesso. Se esistono pool per un ruolo, espande tutte
    le combinazioni simmetriche tra i reparti del pool.
    """

    allowed_roles = _resolve_allowed_roles(
        defaults, fallback_roles=employees_df["ruolo"].unique()
    )
    allowed_departments = _resolve_allowed_departments(defaults)

    combos: list[tuple[str, str, str]] = []
    seen = set()

    def add(role: str, dept_home: str, dept_target: str):
        key = (role, dept_home, dept_target)
        if key not in seen:
            seen.add(key)
            combos.append(key)

    for role in allowed_roles:
        for dept in allowed_departments:
            add(role, dept, dept)

    if not pools_df.empty:
        pools_by_role = {role: grp for role, grp in pools_df.groupby("ruolo")}
        for role in allowed_roles:
            role_df = pools_by_role.get(role)
            if role_df is None:
                continue
            for _, pool_df in role_df.groupby("pool_id"):
                pool_departments = list(dict.fromkeys(pool_df["reparto"].tolist()))
                for dept_home in pool_departments:
                    for dept_target in pool_departments:
                        add(role, dept_home, dept_target)

    compat_df = pd.DataFrame(
        combos, columns=["ruolo", "reparto_home", "reparto_target"]
    )
    if not compat_df.empty:
        compat_df = compat_df.sort_values(
            ["ruolo", "reparto_home", "reparto_target"]
        ).reset_index(drop=True)
    return compat_df


def load_month_plan(path: str, shifts_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data","turno","codice"}, "month_plan.csv")
    df["data"] = df["data"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    df["codice"] = df["codice"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"month_plan.csv: formato data non valido: {exc}")
    bad_turni = sorted(set(df["turno"].unique()) - {"M","P","N"})
    if bad_turni:
        raise LoaderError(f"month_plan.csv: turni non ammessi: {bad_turni}. Ammessi: ['M','P','N']")
    missing_in_catalog = sorted(set(df["turno"].unique()) - set(shifts_df["shift_id"].unique()))
    if missing_in_catalog:
        raise LoaderError(f"month_plan.csv: turni assenti dal catalogo shifts.csv: {missing_in_catalog}")
    return df

def load_coverage_groups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"codice","turno","gruppo","total_min","ruoli_totale"}, "coverage_groups.csv")
    df["total_min"] = pd.to_numeric(df["total_min"], errors="raise").astype(int)
    def _split_roles(s):
        return [x.strip() for x in str(s).split("|") if x.strip()]
    df["ruoli_totale_list"] = df["ruoli_totale"].apply(_split_roles)
    if df.duplicated(subset=["codice","turno","gruppo"]).any():
        dup = df[df.duplicated(subset=["codice","turno","gruppo"], keep=False)].sort_values(["codice","turno","gruppo"])
        raise LoaderError(f"coverage_groups.csv: duplicati su (codice,turno,gruppo):\n{dup}")
    if df["ruoli_totale_list"].apply(len).eq(0).any():
        bad = df[df["ruoli_totale_list"].apply(len).eq(0)]
        raise LoaderError(f"coverage_groups.csv: ruoli_totale vuoto per righe:\n{bad}")
    return df

def load_coverage_roles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"codice","turno","gruppo","ruolo","min_ruolo"}, "coverage_roles.csv")
    df["min_ruolo"] = pd.to_numeric(df["min_ruolo"], errors="raise").astype(int)
    if df.duplicated(subset=["codice","turno","gruppo","ruolo"]).any():
        dup = df[df.duplicated(subset=["codice","turno","gruppo","ruolo"], keep=False)].sort_values(["codice","turno","gruppo","ruolo"])
        raise LoaderError(f"coverage_roles.csv: duplicati su (codice,turno,gruppo,ruolo):\n{dup}")
    return df

def validate_groups_roles(groups: pd.DataFrame, roles: pd.DataFrame, eligibility_df: pd.DataFrame):
    """
    Validazioni incrociate tra coverage_groups, coverage_roles e idoneità turno↔ruolo.

    Requisiti imposti:
      1) Ogni riga di coverage_roles deve riferirsi a un (codice, turno, gruppo) esistente in coverage_groups.
      2) Ogni (turno, ruolo) in coverage_roles deve essere idoneo secondo shift_role_eligibility.
      3) Ogni ruolo elencato in ruoli_totale (coverage_groups) deve essere idoneo per quel turno.
      4) Ogni ruolo in coverage_roles deve essere un sottoinsieme di ruoli_totale_list del gruppo corrispondente.
      5) Per ciascun (codice, turno, gruppo): total_min >= somma(min_ruolo).
    """
    # 1) coverage_roles deve riferirsi a gruppi esistenti
    r_join = roles.merge(
        groups[["codice", "turno", "gruppo", "ruoli_totale_list"]],
        on=["codice", "turno", "gruppo"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    missing_grp = r_join[r_join["_merge"] == "left_only"][["codice", "turno", "gruppo", "ruolo"]].drop_duplicates()
    if not missing_grp.empty:
        raise LoaderError(
            "coverage_roles.csv: (codice,turno,gruppo) non definito in coverage_groups per:\n"
            f"{missing_grp}"
        )

    # 2) coverage_roles: (turno,ruolo) deve essere idoneo
    er = roles.merge(
        eligibility_df.rename(columns={"shift_id": "turno"}),
        on=["turno", "ruolo"],
        how="left",
        indicator=True,
    )
    bad = er[er["_merge"] == "left_only"][["codice", "turno", "gruppo", "ruolo"]].drop_duplicates()
    if not bad.empty:
        raise LoaderError(
            "coverage_roles.csv: (turno,ruolo) non idoneo secondo shift_role_eligibility:\n"
            f"{bad}"
        )

    # 3) coverage_groups: ogni ruolo in ruoli_totale deve essere idoneo per quel turno
    rows = []
    for _, g in groups.iterrows():
        for ruolo in g["ruoli_totale_list"]:
            rows.append((g["codice"], g["turno"], g["gruppo"], ruolo))
    if rows:
        total_roles_df = pd.DataFrame(rows, columns=["codice", "turno", "gruppo", "ruolo"])
        tr = total_roles_df.merge(
            eligibility_df.rename(columns={"shift_id": "turno"}),
            on=["turno", "ruolo"],
            how="left",
            indicator=True,
        )
        bad2 = tr[tr["_merge"] == "left_only"][["codice", "turno", "gruppo", "ruolo"]].drop_duplicates()
        if not bad2.empty:
            raise LoaderError(
                "coverage_groups.csv: ruoli_totale include ruoli non idonei per il turno:\n"
                f"{bad2}"
            )

    # 4) coverage_roles ⊆ ruoli_totale_list del gruppo
    not_in_set = r_join[~r_join.apply(lambda r: r["ruolo"] in set(r["ruoli_totale_list"] or []), axis=1)]
    if not not_in_set.empty:
        raise LoaderError(
            "coverage_roles.csv: ruoli non inclusi in ruoli_totale_list del gruppo corrispondente:\n"
            f"{not_in_set[['codice','turno','gruppo','ruolo']].drop_duplicates()}"
        )

    # 5) total_min >= somma dei minimi per ruolo
    sums = roles.groupby(["codice", "turno", "gruppo"], as_index=False)["min_ruolo"] \
                .sum().rename(columns={"min_ruolo": "sum_min_ruolo"})
    chk = groups.merge(sums, on=["codice", "turno", "gruppo"], how="left").fillna({"sum_min_ruolo": 0})
    viol = chk[chk["total_min"] < chk["sum_min_ruolo"]]
    if not viol.empty:
        raise LoaderError(
            "Incoerenza: total_min < somma(min_ruolo) per:\n"
            f"{viol[['codice','turno','gruppo','total_min','sum_min_ruolo']]}"
        )


def expand_requirements(month_plan: pd.DataFrame, groups: pd.DataFrame, roles: pd.DataFrame):
    base_cols = ["data", "turno", "codice"]
    optional_cols = [c for c in ["data_dt"] if c in month_plan.columns]
    month_plan_base = month_plan[base_cols + optional_cols]

    gt = month_plan_base.merge(groups, on=["codice","turno"], how="left", validate="many_to_many")
    if gt["gruppo"].isna().any():
        miss = gt[gt["gruppo"].isna()].drop_duplicates(subset=["codice","turno"])[["codice","turno"]]
        raise LoaderError(f"month_plan contiene (codice,turno) senza definizione in coverage_groups:\n{miss}")
    gt["ruoli_totale_set"] = gt["ruoli_totale_list"].apply(lambda xs: "|".join(xs))
    ordered_cols = ["data"] + optional_cols + ["turno", "codice", "gruppo", "total_min", "ruoli_totale_set"]
    gt = gt[ordered_cols].sort_values(["data", "turno", "codice", "gruppo"])

    gr = month_plan_base.merge(roles, on=["codice","turno"], how="left", validate="many_to_many")
    gr = gr.dropna(subset=["gruppo"], how="any")
    ordered_cols_roles = ["data"] + optional_cols + ["turno", "codice", "gruppo", "ruolo", "min_ruolo"]
    gr = gr[ordered_cols_roles].sort_values(["data", "turno", "codice", "gruppo", "ruolo"])
    return gt.reset_index(drop=True), gr.reset_index(drop=True)

def attach_calendar(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        cal[[
            "data",
            "dow_iso",
            "week_start_date",
            "week_start_date_dt",
            "week_id",
            "week_idx",
            "is_in_horizon",
            "is_weekend",
            "is_weekday_holiday",
            "holiday_desc",
        ]],
        on="data",
        how="left",
    )

def load_availability(
    path: str,
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    availability.csv supporta:
      A) per-turno: colonne {data, employee_id, turno} con turno in ALLOWED_TURNS (derivati da shifts_df con duration_min>0)
      B) tutto il giorno: colonne {data, employee_id} (senza 'turno') oppure 'turno' vuoto/'ALL'/'*'
         -> espansa automaticamente in una riga per ciascun turno in ALLOWED_TURNS.

    Restituisce: [data, employee_id, turno, dow_iso, week_id, week_idx, is_in_horizon,
    indicatori weekend/festivi e orari turno]
    """
    import os

    # Turni consentiti dinamicamente dal catalogo: tutti i turni "di lavoro" (durata > 0)
    allowed_turns = tuple(
        pd.Series(shifts_df.loc[shifts_df["duration_min"] > 0, "shift_id"])
          .astype(str).str.strip().unique().tolist()
    )
    if not allowed_turns:
        raise LoaderError("shifts.csv: nessun turno con duration_min>0 (niente turni lavorativi disponibili).")

    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "data",
                "data_dt",
                "employee_id",
                "turno",
                "shift_start_time",
                "shift_end_time",
                "shift_start_dt",
                "shift_end_dt",
                "shift_duration_min",
                "shift_crosses_midnight",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        )

    df = pd.read_csv(path, dtype=str).fillna("")
    # Richiede sempre almeno data, employee_id
    _ensure_cols(df, {"data","employee_id"}, "availability.csv")

    # Normalizza
    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"availability.csv: formato data non valido: {exc}")

    # Se 'turno' non esiste, creala vuota (interpretabile come ALL-day)
    if "turno" not in df.columns:
        df["turno"] = ""
    df["turno"] = df["turno"].astype(str).str.strip().str.upper()

    # Verifica dipendenti noti
    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"availability.csv: employee_id sconosciuti: {unknown}")

    # Alias per "tutto il giorno"
    allday_mask = (df["turno"] == "") | (df["turno"] == "ALL") | (df["turno"] == "*")
    perturno_mask = ~allday_mask

    # Valida i turni forniti esplicitamente contro il catalogo
    bad_turni = sorted(set(df.loc[perturno_mask, "turno"].unique()) - set(allowed_turns))
    if bad_turni:
        raise LoaderError(
            f"availability.csv: turni non ammessi: {bad_turni}. "
            f"Ammessi (da shifts.csv con duration_min>0): {sorted(allowed_turns)} "
            f"oppure ALL/*/vuoto per indisponibilità tutto il giorno."
        )

    # Espansione: righe ALL-day -> tutte le voci in allowed_turns
    rows = []
    for _, r in df.iterrows():
        base_row = {"data": r["data"], "data_dt": r["data_dt"], "employee_id": r["employee_id"]}
        t = r["turno"]
        if t in allowed_turns:
            rows.append({**base_row, "turno": t})
        else:
            for tt in allowed_turns:
                rows.append({**base_row, "turno": tt})

    out = pd.DataFrame(rows).drop_duplicates(subset=["data", "employee_id", "turno"]).reset_index(drop=True)

    # Aggancio al calendario e vincolo orizzonte
    out = attach_calendar(out, calendar_df)
    if (~out["is_in_horizon"].astype(bool)).any():
        outside = out[~out["is_in_horizon"].astype(bool)][["data","employee_id","turno"]]
        raise LoaderError(f"availability.csv: presenti righe fuori orizzonte:\n{outside.head()}")

    shift_cols = [
        "shift_id",
        "start_time",
        "end_time",
        "duration_min",
        "crosses_midnight",
    ]
    shift_info = shifts_df[shift_cols].rename(
        columns={
            "shift_id": "turno",
            "start_time": "shift_start_time",
            "end_time": "shift_end_time",
            "duration_min": "shift_duration_min",
            "crosses_midnight": "shift_crosses_midnight",
        }
    )
    out = out.merge(shift_info, on="turno", how="left", validate="many_to_one")

    out["shift_start_dt"] = out["data_dt"] + out["shift_start_time"]
    out["shift_end_dt"] = out["data_dt"] + out["shift_end_time"]
    crosses_mask = out["shift_crosses_midnight"].fillna(0).astype(int) == 1
    out.loc[crosses_mask, "shift_end_dt"] = out.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)

    return out[[
        "data",
        "data_dt",
        "employee_id",
        "turno",
        "shift_start_time",
        "shift_end_time",
        "shift_start_dt",
        "shift_end_dt",
        "shift_duration_min",
        "shift_crosses_midnight",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]]


def load_leaves(
    path: str,
    employees_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica leaves.csv, espande le assenze su turni lavorativi e produce l'aggregato day-level."""

    allowed_turns = tuple(
        pd.Series(shifts_df.loc[shifts_df["duration_min"] > 0, "shift_id"])
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not allowed_turns:
        raise LoaderError(
            "shifts.csv: nessun turno con duration_min>0 — impossibile espandere leaves.csv."
        )

    shift_columns = [
        "employee_id",
        "data",
        "data_dt",
        "turno",
        "tipo",
        "shift_start_time",
        "shift_end_time",
        "shift_start_dt",
        "shift_end_dt",
        "shift_duration_min",
        "shift_crosses_midnight",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]

    day_columns = [
        "employee_id",
        "data",
        "data_dt",
        "tipo_set",
        "is_leave_day",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]

    if not os.path.exists(path):
        empty_shift = pd.DataFrame(columns=shift_columns)
        empty_day = pd.DataFrame(columns=day_columns)
        return empty_shift, empty_day

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"employee_id", "start_date", "end_date", "tipo"}, "leaves.csv")

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["start_date"] = df["start_date"].astype(str).str.strip()
    df["end_date"] = df["end_date"].astype(str).str.strip()
    df["tipo"] = df["tipo"].astype(str).str.strip()

    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"leaves.csv: employee_id sconosciuti: {unknown}")

    try:
        df["start_date_dt"] = pd.to_datetime(
            df["start_date"], format="%Y-%m-%d", errors="raise"
        )
        df["end_date_dt"] = pd.to_datetime(
            df["end_date"], format="%Y-%m-%d", errors="raise"
        )
    except ValueError as exc:
        raise LoaderError(f"leaves.csv: formato data non valido: {exc}")

    bad_interval = df["start_date_dt"] > df["end_date_dt"]
    if bad_interval.any():
        bad_rows = df.loc[bad_interval, ["employee_id", "start_date", "end_date"]]
        raise LoaderError(
            "leaves.csv: intervallo con start_date > end_date per le righe:\n"
            f"{bad_rows}"
        )

    shift_info = (
        shifts_df.loc[
            shifts_df["shift_id"].isin(allowed_turns),
            ["shift_id", "start_time", "end_time", "crosses_midnight"],
        ]
        .copy()
    )
    shift_rows = list(shift_info.itertuples(index=False))

    records = []
    day_records = []
    for row in df.itertuples(index=False):
        absence_start_day = row.start_date_dt.normalize()
        absence_end_day = row.end_date_dt.normalize()
        absence_interval_start = absence_start_day
        absence_interval_end = absence_end_day + pd.Timedelta(days=1)

        current_day = absence_start_day
        while current_day <= absence_end_day:
            day_records.append(
                {
                    "employee_id": row.employee_id,
                    "data": current_day.date().isoformat(),
                    "tipo": row.tipo,
                }
            )
            current_day += pd.Timedelta(days=1)

        day = absence_start_day - pd.Timedelta(days=1)
        last_day = absence_end_day

        while day <= last_day:
            day_str = day.date().isoformat()
            for shift in shift_rows:
                if pd.isna(shift.start_time) or pd.isna(shift.end_time):
                    continue

                shift_start_dt = day + shift.start_time
                shift_end_dt = day + shift.end_time
                if int(shift.crosses_midnight) == 1:
                    shift_end_dt = shift_end_dt + pd.Timedelta(days=1)

                if shift_end_dt > absence_interval_start and shift_start_dt < absence_interval_end:
                    records.append(
                        {
                            "employee_id": row.employee_id,
                            "data": day_str,
                            "turno": shift.shift_id,
                            "tipo": row.tipo,
                        }
                    )
            day += pd.Timedelta(days=1)

    if records:
        shift_out = pd.DataFrame.from_records(records)
        shift_out["data_dt"] = pd.to_datetime(shift_out["data"], format="%Y-%m-%d")

        shift_out = shift_out.drop_duplicates(
            subset=["employee_id", "data", "turno", "tipo"]
        ).reset_index(drop=True)

        shift_out = attach_calendar(shift_out, calendar_df)
        shift_out = shift_out[shift_out["is_in_horizon"].fillna(False)].copy()

        shift_cols = [
        "shift_id",
        "start_time",
        "end_time",
        "duration_min",
        "crosses_midnight",
    ]
        shift_info = shifts_df[shift_cols].rename(
            columns={
                "shift_id": "turno",
                "start_time": "shift_start_time",
                "end_time": "shift_end_time",
                "duration_min": "shift_duration_min",
                "crosses_midnight": "shift_crosses_midnight",
            }
        )

        shift_out = shift_out.merge(shift_info, on="turno", how="left", validate="many_to_one")

        shift_out["shift_start_dt"] = shift_out["data_dt"] + shift_out["shift_start_time"]
        shift_out["shift_end_dt"] = shift_out["data_dt"] + shift_out["shift_end_time"]
        crosses_mask = shift_out["shift_crosses_midnight"].fillna(0).astype(int) == 1
        shift_out.loc[crosses_mask, "shift_end_dt"] = (
            shift_out.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)
        )

        shift_out = shift_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "turno",
                "tipo",
                "shift_start_time",
                "shift_end_time",
                "shift_start_dt",
                "shift_end_dt",
                "shift_duration_min",
                "shift_crosses_midnight",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        ].sort_values(["data", "employee_id", "turno"]).reset_index(drop=True)
    else:
        shift_out = pd.DataFrame(columns=shift_columns)

    if day_records:
        day_out = pd.DataFrame.from_records(day_records)
        day_out = day_out.drop_duplicates().reset_index(drop=True)
        day_out["data_dt"] = pd.to_datetime(day_out["data"], format="%Y-%m-%d")
        day_out = attach_calendar(day_out, calendar_df)
        day_out = day_out[day_out["is_in_horizon"].fillna(False)].copy()

        def _join_types(values: pd.Series) -> str:
            unique_vals = sorted({str(v).strip() for v in values if str(v).strip()})
            return "|".join(unique_vals)

        day_out = (
            day_out.groupby(["employee_id", "data"], as_index=False)
            .agg(
                data_dt=("data_dt", "first"),
                tipo_set=("tipo", _join_types),
                dow_iso=("dow_iso", "first"),
                week_start_date=("week_start_date", "first"),
                week_start_date_dt=("week_start_date_dt", "first"),
                week_id=("week_id", "first"),
                week_idx=("week_idx", "first"),
                is_in_horizon=("is_in_horizon", "first"),
                is_weekend=("is_weekend", "first"),
                is_weekday_holiday=("is_weekday_holiday", "first"),
                holiday_desc=("holiday_desc", "first"),
            )
        )
        day_out["is_leave_day"] = 1
        day_out = day_out[
            [
                "employee_id",
                "data",
                "data_dt",
                "tipo_set",
                "is_leave_day",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        ].sort_values(["data", "employee_id"]).reset_index(drop=True)
    else:
        day_out = pd.DataFrame(columns=day_columns)

    return shift_out, day_out


def load_history(
    path: str,
    employees_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Carica lo storico assegnazioni (history.csv) con validazioni aggiuntive:
      - colonne obbligatorie: data, employee_id, turno
      - formato data: YYYY-MM-DD
      - turno deve esistere nel catalogo shifts (ammessi anche R e SN)
      - employee_id deve esistere in anagrafica
      - dedup di record identici (data, employee_id, turno)
      - errore se uno stesso (data, employee_id) compare con più 'turno' diversi (max 1 turno/giorno per dipendente)
      - NON vincola all’orizzonte: può contenere giorni precedenti per i vincoli a cavallo
    Ritorna: [data, employee_id, turno, dow_iso, week_id, week_idx, is_in_horizon,
    indicatori weekend/festivi e metadati del turno]
    """
    import os
    from datetime import datetime

    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "data",
                "data_dt",
                "employee_id",
                "turno",
                "shift_start_time",
                "shift_end_time",
                "shift_start_dt",
                "shift_end_dt",
                "shift_duration_min",
                "shift_crosses_midnight",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        )

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"data","employee_id","turno"}, "history.csv")

    # Normalizza stringhe
    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    try:
        df["data_dt"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="raise")
    except ValueError as exc:
        raise LoaderError(f"history.csv: formato data non valido: {exc}")

    # Formato data YYYY-MM-DD
    def _is_iso_date(s: str) -> bool:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except Exception:
            return False
    bad_date_mask = ~df["data"].apply(_is_iso_date)
    if bad_date_mask.any():
        bad_rows = df.loc[bad_date_mask, ["data","employee_id","turno"]].head()
        raise LoaderError(f"history.csv: formato data non valido (atteso YYYY-MM-DD) per le righe:\n{bad_rows}")

    # Turni esistenti nel catalogo
    known_shifts = set(shifts_df["shift_id"].astype(str).str.strip().unique())
    bad_turni = sorted(set(df["turno"].unique()) - known_shifts)
    if bad_turni:
        raise LoaderError(f"history.csv: turni non presenti in shifts.csv: {bad_turni}")

    # Employee esistenti
    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"history.csv: employee_id sconosciuti rispetto a employees.csv: {unknown}")

    # Dedup record identici
    df = df.drop_duplicates(subset=["data","employee_id","turno"]).reset_index(drop=True)

    # Conflitti: stesso (data, employee_id) con più turni diversi -> errore
    conflicts = (
        df.groupby(["data","employee_id"])["turno"]
          .nunique()
          .reset_index(name="n_turni")
    )
    conflicts = conflicts[conflicts["n_turni"] > 1]
    if not conflicts.empty:
        # estrai esempi concreti delle righe in conflitto
        keys = set(map(tuple, conflicts[["data","employee_id"]].to_records(index=False)))
        sample = df[df.set_index(["data","employee_id"]).index.isin(keys)].sort_values(["data","employee_id","turno"]).head(20)
        raise LoaderError(
            "history.csv: più di un turno per lo stesso dipendente nello stesso giorno (max 1 turno/giorno). "
            "Esempi delle righe in conflitto:\n"
            f"{sample[['data','employee_id','turno']]}"
        )

    # Aggancio al calendario (può includere giorni fuori orizzonte: è voluto)
    df = attach_calendar(df, calendar_df)

    shift_cols = [
        "shift_id",
        "start_time",
        "end_time",
        "duration_min",
        "crosses_midnight",
    ]
    shift_info = shifts_df[shift_cols].rename(
        columns={
            "shift_id": "turno",
            "start_time": "shift_start_time",
            "end_time": "shift_end_time",
            "duration_min": "shift_duration_min",
            "crosses_midnight": "shift_crosses_midnight",
        }
    )
    df = df.merge(shift_info, on="turno", how="left", validate="many_to_one")

    df["shift_start_dt"] = df["data_dt"] + df["shift_start_time"]
    df["shift_end_dt"] = df["data_dt"] + df["shift_end_time"]
    crosses_mask = df["shift_crosses_midnight"].fillna(0).astype(int) == 1
    df.loc[crosses_mask, "shift_end_dt"] = df.loc[crosses_mask, "shift_end_dt"] + pd.Timedelta(days=1)

    return df[[
        "data",
        "data_dt",
        "employee_id",
        "turno",
        "shift_start_time",
        "shift_end_time",
        "shift_start_dt",
        "shift_end_dt",
        "shift_duration_min",
        "shift_crosses_midnight",
        "dow_iso",
        "week_start_date",
        "week_start_date_dt",
        "week_id",
        "week_idx",
        "is_in_horizon",
        "is_weekend",
        "is_weekday_holiday",
        "holiday_desc",
    ]]


def load_all(config_path: str, data_dir: str) -> LoadedData:
    cfg = load_config(config_path)
    start_date = _parse_date(cfg["horizon"]["start_date"])
    end_date = _parse_date(cfg["horizon"]["end_date"])

    defaults = cfg.get("defaults", {})

    holidays_df = load_holidays(os.path.join(data_dir, "holidays.csv"))

    calendar_df = build_calendar(start_date, end_date, holidays_df if not holidays_df.empty else None)

    employees_df = load_employees(os.path.join(data_dir, "employees.csv"), defaults)
    shifts_df = load_shifts(os.path.join(data_dir, "shifts.csv"))
    eligibility_df = load_shift_role_eligibility(os.path.join(data_dir, "shift_role_eligibility.csv"), employees_df, shifts_df)
    role_dept_pools_df = load_role_dept_pools(
        os.path.join(data_dir, "role_dept_pools.csv"),
        defaults,
        employees_df,
    )
    dept_compat_df = build_department_compatibility(
        defaults,
        role_dept_pools_df,
        employees_df,
    )
    month_plan_df = load_month_plan(os.path.join(data_dir, "month_plan.csv"), shifts_df)
    groups_df = load_coverage_groups(os.path.join(data_dir, "coverage_groups.csv"))
    roles_df = load_coverage_roles(os.path.join(data_dir, "coverage_roles.csv"))
    validate_groups_roles(groups_df, roles_df, eligibility_df)

    month_plan_df = attach_calendar(month_plan_df, calendar_df)

    groups_total_expanded, groups_role_min_expanded = expand_requirements(month_plan_df, groups_df, roles_df)
    groups_total_expanded = attach_calendar(groups_total_expanded, calendar_df)
    groups_role_min_expanded = attach_calendar(groups_role_min_expanded, calendar_df)

    history_df = load_history(
        os.path.join(data_dir, "history.csv"),
        employees_df,
        shifts_df,
        calendar_df,
    )
    leaves_df, leaves_days_df = load_leaves(
        os.path.join(data_dir, "leaves.csv"),
        employees_df,
        shifts_df,
        calendar_df,
    )
    availability_df = load_availability(
        os.path.join(data_dir, "availability.csv"),
        employees_df,
        calendar_df,
        shifts_df,
    )

    return LoadedData(
        cfg=cfg,
        calendar_df=calendar_df,
        employees_df=employees_df,
        shifts_df=shifts_df,
        eligibility_df=eligibility_df,
        month_plan_df=month_plan_df,
        groups_total_expanded=groups_total_expanded,
        groups_role_min_expanded=groups_role_min_expanded,
        history_df=history_df,
        availability_df=availability_df,
        leaves_df=leaves_df,
        leaves_days_df=leaves_days_df,
        holidays_df=holidays_df,
        role_dept_pools_df=role_dept_pools_df,
        dept_compat_df=dept_compat_df,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Loader clinica — Step A (v6)")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data-dir", type=str, default=".")
    ap.add_argument("--export-csv", action="store_true", help="Esporta i DF espansi come CSV di debug")
    args = ap.parse_args()

    data = load_all(args.config, args.data_dir)

    print("OK: caricati e costruiti i dati.")
    print(f"- employees: {len(data.employees_df)}")
    print(f"- month_plan righe: {len(data.month_plan_df)}")
    print(f"- coverage groups (espansi): {len(data.groups_total_expanded)}")
    print(f"- coverage role min (espansi): {len(data.groups_role_min_expanded)}")
    print(f"- calendar giorni: {len(data.calendar_df)}")
    print(f"- availability righe: {len(data.availability_df)}")
    print(f"- leaves righe: {len(data.leaves_df)}")
    print(f"- leave days: {len(data.leaves_days_df)}")
    print(f"- history righe: {len(data.history_df)}")
    print(f"- eligibility coppie (turno,ruolo): {len(data.eligibility_df)}")
    if not data.holidays_df.empty:
        print(f"- holidays caricati: {len(data.holidays_df)}")

    if args.export_csv:
        outdir = os.path.join(args.data_dir, "_expanded")
        os.makedirs(outdir, exist_ok=True)
        data.calendar_df.to_csv(os.path.join(outdir, "calendar.csv"), index=False)
        data.month_plan_df.to_csv(os.path.join(outdir, "month_plan_with_calendar.csv"), index=False)
        data.groups_total_expanded.to_csv(os.path.join(outdir, "groups_total_expanded.csv"), index=False)
        data.groups_role_min_expanded.to_csv(os.path.join(outdir, "groups_role_min_expanded.csv"), index=False)
        data.employees_df.to_csv(os.path.join(outdir, "employees_processed.csv"), index=False)
        data.shifts_df.to_csv(os.path.join(outdir, "shifts_processed.csv"), index=False)
        data.availability_df.to_csv(os.path.join(outdir, "availability_with_calendar.csv"), index=False)
        data.leaves_df.to_csv(os.path.join(outdir, "leaves_expanded.csv"), index=False)
        data.leaves_days_df.to_csv(os.path.join(outdir, "leaves_days.csv"), index=False)
        data.history_df.to_csv(os.path.join(outdir, "history_with_calendar.csv"), index=False)
        data.eligibility_df.to_csv(os.path.join(outdir, "shift_role_eligibility_processed.csv"), index=False)
        print(f"Esportati CSV di debug in: {outdir}")
