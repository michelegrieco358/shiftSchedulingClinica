
"""
Loader clinica — 
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Set
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

def _parse_date(s: str) -> date:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()

def _ensure_cols(df: pd.DataFrame, required: Set[str], label: str):
    missing = required - set(df.columns)
    if missing:
        raise LoaderError(f"{label}: colonne mancanti {sorted(missing)}")

def build_calendar(start_date: date, end_date: date) -> pd.DataFrame:
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
            - dow_iso: Giorno della settimana ISO (1=lunedì, 7=domenica)
            - week_start_date: Data di inizio settimana (lunedì) in formato ISO
            - week_id: Identificativo della settimana (uguale a week_start_date)
            - is_in_horizon: Boolean che indica se la data è nel periodo di pianificazione
            - week_idx: Indice numerico progressivo della settimana (0, 1, 2, ...)
            - cal_start: Data di inizio del calendario esteso
            
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

def load_employees(path: str, defaults: dict) -> pd.DataFrame:
    """
    Legge employees.csv e applica:
    - unicità employee_id
    - validazione ruoli rispetto a defaults['allowed_roles'] (se presente)
    - ore dovute: se vuote, usa defaults['contract_hours_by_role_h'][ruolo]
    - divieto di valori negativi (eccetto saldo iniziale che può essere < 0)
    - conversione ore → minuti per i campi rilevanti
    """
    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(
        df,
        {"employee_id", "nome", "ruolo", "ore_dovute_mese_h", "saldo_prog_iniziale_h"},
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
    allowed_roles_cfg = defaults.get("allowed_roles", None)
    if isinstance(allowed_roles_cfg, str):
        allowed_roles = [x.strip() for x in allowed_roles_cfg.replace(",", "|").split("|") if x.strip()]
    elif isinstance(allowed_roles_cfg, (list, tuple, set)):
        allowed_roles = [str(x).strip() for x in allowed_roles_cfg if str(x).strip()]
    else:
        # Fallback: accetta i ruoli presenti nel file (non bloccante)
        allowed_roles = sorted(df["ruolo"].unique())

    bad_roles = sorted(set(df["ruolo"].unique()) - set(allowed_roles))
    if bad_roles:
        raise LoaderError(f"employees.csv: ruoli non ammessi rispetto alla config: {bad_roles}")

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

    # --- Ritorna solo le colonne utili ---
    return df[
        [
            "employee_id",
            "nome",
            "ruolo",
            "dovuto_min",
            "saldo_init_min",
            "max_week_min",
            "max_month_extra_min",
            "max_nights_week",
            "max_nights_month",
        ]
    ]


def load_shifts(path: str) -> pd.DataFrame:
    """
    Carica e valida il catalogo turni (shifts.csv).
    Requisiti:
      - colonne obbligatorie: shift_id, start, end, duration_min, crosses_midnight
      - shift_id univoci (duplicati identici vengono deduplicati; se divergenti -> errore)
      - R e SN: duration_min=0, crosses_midnight=0, start/end vuoti
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

    # Regole speciali per R/SN (riposo, smonto notte)
    mask_zero = df["shift_id"].isin({"R", "SN"})
    if not df.loc[mask_zero, "duration_min"].eq(0).all() or not df.loc[mask_zero, "crosses_midnight"].eq(0).all():
        raise LoaderError("shifts.csv: R e SN devono avere duration_min=0 e crosses_midnight=0")

    # Per R/SN richiediamo start/end vuoti (stringa vuota)
    if (df.loc[mask_zero, ["start", "end"]] != "").any().any():
        raise LoaderError("shifts.csv: R e SN devono avere start/end vuoti")

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

    return df[["shift_id", "start", "end", "duration_min", "crosses_midnight"]]


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


def load_month_plan(path: str, shifts_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data","turno","codice"}, "month_plan.csv")
    df["data"] = df["data"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    df["codice"] = df["codice"].astype(str).str.strip()
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
    gt = month_plan.merge(groups, on=["codice","turno"], how="left", validate="many_to_many")
    if gt["gruppo"].isna().any():
        miss = gt[gt["gruppo"].isna()].drop_duplicates(subset=["codice","turno"])[["codice","turno"]]
        raise LoaderError(f"month_plan contiene (codice,turno) senza definizione in coverage_groups:\n{miss}")
    gt["ruoli_totale_set"] = gt["ruoli_totale_list"].apply(lambda xs: "|".join(xs))
    gt = gt[["data","turno","codice","gruppo","total_min","ruoli_totale_set"]].sort_values(["data","turno","codice","gruppo"])
    gr = month_plan.merge(roles, on=["codice","turno"], how="left", validate="many_to_many")
    gr = gr.dropna(subset=["gruppo"], how="any")
    gr = gr[["data","turno","codice","gruppo","ruolo","min_ruolo"]].sort_values(["data","turno","codice","gruppo","ruolo"])
    return gt.reset_index(drop=True), gr.reset_index(drop=True)

def attach_calendar(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    return df.merge(cal[["data","dow_iso","week_id","week_idx","is_in_horizon"]], on="data", how="left")

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

    Restituisce: [data, employee_id, turno, dow_iso, week_id, week_idx, is_in_horizon]
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
        return pd.DataFrame(columns=["data","employee_id","turno","dow_iso","week_id","week_idx","is_in_horizon"])

    df = pd.read_csv(path, dtype=str).fillna("")
    # Richiede sempre almeno data, employee_id
    _ensure_cols(df, {"data","employee_id"}, "availability.csv")

    # Normalizza
    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()

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
        data = r["data"]
        emp = r["employee_id"]
        t = r["turno"]
        if t in allowed_turns:
            rows.append((data, emp, t))
        else:
            for tt in allowed_turns:
                rows.append((data, emp, tt))

    out = pd.DataFrame(rows, columns=["data","employee_id","turno"]).drop_duplicates().reset_index(drop=True)

    # Aggancio al calendario e vincolo orizzonte
    out = attach_calendar(out, calendar_df)
    if (~out["is_in_horizon"].astype(bool)).any():
        outside = out[~out["is_in_horizon"].astype(bool)][["data","employee_id","turno"]]
        raise LoaderError(f"availability.csv: presenti righe fuori orizzonte:\n{outside.head()}")

    return out


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
    Ritorna: [data, employee_id, turno, dow_iso, week_id, week_idx, is_in_horizon]
    """
    import os
    from datetime import datetime

    if not os.path.exists(path):
        return pd.DataFrame(columns=["data","employee_id","turno","dow_iso","week_id","week_idx","is_in_horizon"])

    df = pd.read_csv(path, dtype=str).fillna("")

    _ensure_cols(df, {"data","employee_id","turno"}, "history.csv")

    # Normalizza stringhe
    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()

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

    return df[["data","employee_id","turno","dow_iso","week_id","week_idx","is_in_horizon"]]


def load_all(config_path: str, data_dir: str) -> LoadedData:
    cfg = load_config(config_path)
    start_date = _parse_date(cfg["horizon"]["start_date"])
    end_date = _parse_date(cfg["horizon"]["end_date"])

    calendar_df = build_calendar(start_date, end_date)

    employees_df = load_employees(os.path.join(data_dir, "employees.csv"), cfg.get("defaults", {}))
    shifts_df = load_shifts(os.path.join(data_dir, "shifts.csv"))
    eligibility_df = load_shift_role_eligibility(os.path.join(data_dir, "shift_role_eligibility.csv"), employees_df, shifts_df)
    month_plan_df = load_month_plan(os.path.join(data_dir, "month_plan.csv"), shifts_df)
    groups_df = load_coverage_groups(os.path.join(data_dir, "coverage_groups.csv"))
    roles_df = load_coverage_roles(os.path.join(data_dir, "coverage_roles.csv"))
    validate_groups_roles(groups_df, roles_df, eligibility_df)

    month_plan_df = attach_calendar(month_plan_df, calendar_df)

    groups_total_expanded, groups_role_min_expanded = expand_requirements(month_plan_df[["data","turno","codice"]], groups_df, roles_df)
    groups_total_expanded = attach_calendar(groups_total_expanded, calendar_df)
    groups_role_min_expanded = attach_calendar(groups_role_min_expanded, calendar_df)

    history_df = load_history(os.path.join(data_dir, "history.csv"), shifts_df, calendar_df)
    availability_df = load_availability(os.path.join(data_dir, "availability.csv"), employees_df, calendar_df)

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
    print(f"- history righe: {len(data.history_df)}")
    print(f"- eligibility coppie (turno,ruolo): {len(data.eligibility_df)}")

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
        data.history_df.to_csv(os.path.join(outdir, "history_with_calendar.csv"), index=False)
        data.eligibility_df.to_csv(os.path.join(outdir, "shift_role_eligibility_processed.csv"), index=False)
        print(f"Esportati CSV di debug in: {outdir}")
