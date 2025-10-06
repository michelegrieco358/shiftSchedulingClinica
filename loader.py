
# -*- coding: utf-8 -*-
"""
Loader clinica — Step A (v2, con availability)
- Settimana ISO (lunedì).
- Calendario da min(prev_week_start, start_date-6 giorni) a end_date.
- Availability: ogni riga (data, employee_id, turno) è una indisponibilità HARD.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Set
import pandas as pd
import yaml
from datetime import datetime, date, timedelta

TURNI_DOMANDA: Set[str] = {"M", "P", "N"}  # R/SN non nella domanda

class LoaderError(Exception):
    pass

@dataclass
class LoadedData:
    cfg: dict
    calendar_df: pd.DataFrame
    employees_df: pd.DataFrame
    shifts_df: pd.DataFrame
    month_plan_df: pd.DataFrame
    groups_total_expanded: pd.DataFrame
    groups_role_min_expanded: pd.DataFrame
    history_df: pd.DataFrame
    availability_df: pd.DataFrame

# ----------------- util -----------------
def _parse_date(s: str) -> date:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()

def _ensure_cols(df: pd.DataFrame, required: Set[str], label: str):
    missing = required - set(df.columns)
    if missing:
        raise LoaderError(f"{label}: colonne mancanti {sorted(missing)}")

# ----------------- calendar -----------------
def build_calendar(start_date: date, end_date: date) -> pd.DataFrame:
    # Monday-based weeks (ISO).
    prev_week_start = start_date - timedelta(days=(start_date.isoweekday() - 1))
    six_days_before = start_date - timedelta(days=6)
    cal_start = min(prev_week_start, six_days_before)
    rows = []
    d = cal_start
    while d <= end_date:
        dow_iso = d.isoweekday()  # 1..7 (Mon=1)
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

# ----------------- loaders -----------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    try:
        cfg_h = cfg["horizon"]
        start_date = _parse_date(cfg_h["start_date"])
        end_date = _parse_date(cfg_h["end_date"])
    except Exception as e:
        raise LoaderError(f"config: horizon/start_date,end_date mancanti o non validi: {e}")
    if end_date < start_date:
        raise LoaderError("config: end_date < start_date")
    return cfg

def load_employees(path: str, defaults: dict) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"employee_id","nome","ruolo","ore_dovute_mese_h","saldo_prog_iniziale_h"}, "employees.csv")

    def _to_min(x):
        x = str(x).strip()
        return int(round(float(x)*60)) if x else 0
    df["dovuto_min"] = df["ore_dovute_mese_h"].apply(_to_min)
    df["saldo_init_min"] = df["saldo_prog_iniziale_h"].apply(_to_min)

    for col, dflt in [
        ("max_week_hours_h", defaults.get("max_week_hours_h", 60)),
        ("max_month_extra_h", defaults.get("max_month_extra_h", 40)),
        ("max_nights_week", defaults.get("max_nights_week", 3)),
        ("max_nights_month", defaults.get("max_nights_month", 8)),
    ]:
        if col in df.columns:
            val = df[col].astype(str).str.strip().replace({"": None})
            df[col] = val.fillna(dflt)
        else:
            df[col] = dflt

    df["max_week_min"] = (pd.to_numeric(df["max_week_hours_h"], errors="coerce").fillna(0)*60).astype(int)
    df["max_month_extra_min"] = (pd.to_numeric(df["max_month_extra_h"], errors="coerce").fillna(0)*60).astype(int)

    return df[["employee_id","nome","ruolo","dovuto_min","saldo_init_min","max_week_min","max_month_extra_min","max_nights_week","max_nights_month"]]

def load_shifts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    _ensure_cols(df, {"shift_id","start","end","duration_min","crosses_midnight"}, "shifts.csv")
    df = df[["shift_id","start","end","duration_min","crosses_midnight"]]
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="raise").astype(int)
    df["crosses_midnight"] = pd.to_numeric(df["crosses_midnight"], errors="raise").astype(int)
    return df

def load_month_plan(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data","turno","codice"}, "month_plan.csv")
    df["data"] = df["data"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    df["codice"] = df["codice"].astype(str).str.strip()
    bad_turni = sorted(set(df["turno"].unique()) - {"M","P","N"})
    if bad_turni:
        raise LoaderError(f"month_plan.csv: turni non ammessi: {bad_turni}. Ammessi: ['M','P','N']")
    return df

def load_coverage_groups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"codice","turno","gruppo","total_min","ruoli_totale"}, "coverage_groups.csv")
    df["total_min"] = pd.to_numeric(df["total_min"], errors="raise").astype(int)
    df["ruoli_totale_list"] = df["ruoli_totale"].apply(lambda s: [x.strip() for x in str(s).split("|") if x.strip()])
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

def validate_groups_roles(groups: pd.DataFrame, roles: pd.DataFrame):
    merged = roles.merge(groups[["codice","turno","gruppo","ruoli_totale_list"]],
                         on=["codice","turno","gruppo"], how="left", validate="many_to_one")
    bad = merged[~merged.apply(lambda r: r["ruolo"] in set(r["ruoli_totale_list"] or []), axis=1)]
    if not bad.empty:
        raise LoaderError(f"coverage_roles.csv: ruoli fuori da ruoli_totale per:\n{bad[['codice','turno','gruppo','ruolo']]}")
    sums = roles.groupby(["codice","turno","gruppo"], as_index=False)["min_ruolo"].sum().rename(columns={"min_ruolo":"sum_min_ruolo"})
    chk = groups.merge(sums, on=["codice","turno","gruppo"], how="left").fillna({"sum_min_ruolo":0})
    viol = chk[ chk["total_min"] < chk["sum_min_ruolo"] ]
    if not viol.empty:
        raise LoaderError(f"Incoerenza: total_min < somma(min_ruolo) per:\n{viol[['codice','turno','gruppo','total_min','sum_min_ruolo']]}")

def expand_requirements(month_plan: pd.DataFrame, groups: pd.DataFrame, roles: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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

def load_availability(path: str, employees_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["data","employee_id","turno","dow_iso","week_id","week_idx","is_in_horizon"])
    df = pd.read_csv(path, dtype=str).fillna("")
    _ensure_cols(df, {"data","employee_id","turno"}, "availability.csv")
    df["data"] = df["data"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["turno"] = df["turno"].astype(str).str.strip()
    bad_turni = sorted(set(df["turno"].unique()) - {"M","P","N"})
    if bad_turni:
        raise LoaderError(f"availability.csv: turni non ammessi: {bad_turni}. Ammessi: ['M','P','N']")
    # employee existence
    known_emp = set(employees_df["employee_id"].unique())
    unknown = sorted(set(df["employee_id"].unique()) - known_emp)
    if unknown:
        raise LoaderError(f"availability.csv: employee_id sconosciuti: {unknown}")
    # deduplicate triplets
    df = df.drop_duplicates(subset=["data","employee_id","turno"]).reset_index(drop=True)
    # attach calendar
    df = attach_calendar(df, calendar_df)
    # ensure only within horizon (hard)
    outside = df[~df["is_in_horizon"].astype(bool)]
    if len(outside) > 0:
        raise LoaderError(f"availability.csv: presenti righe fuori orizzonte pianificato: \n{outside[['data','employee_id','turno']].head()}")
    return df

def load_all(config_path: str, data_dir: str) -> LoadedData:
    cfg = load_config(config_path)
    start_date = _parse_date(cfg["horizon"]["start_date"])
    end_date = _parse_date(cfg["horizon"]["end_date"])

    calendar_df = build_calendar(start_date, end_date)

    employees_df = load_employees(os.path.join(data_dir, "employees.csv"), cfg.get("defaults", {}))
    shifts_df = load_shifts(os.path.join(data_dir, "shifts.csv"))
    month_plan_df = load_month_plan(os.path.join(data_dir, "month_plan.csv"))
    groups_df = load_coverage_groups(os.path.join(data_dir, "coverage_groups.csv"))
    roles_df = load_coverage_roles(os.path.join(data_dir, "coverage_roles.csv"))
    validate_groups_roles(groups_df, roles_df)

    month_plan_df = attach_calendar(month_plan_df, calendar_df)

    groups_total_expanded, groups_role_min_expanded = expand_requirements(month_plan_df[["data","turno","codice"]], groups_df, roles_df)
    groups_total_expanded = attach_calendar(groups_total_expanded, calendar_df)
    groups_role_min_expanded = attach_calendar(groups_role_min_expanded, calendar_df)

    history_df = pd.DataFrame(columns=["data","employee_id","turno","dow_iso","week_id","week_idx","is_in_horizon"])
    hist_path = os.path.join(data_dir, "history.csv")
    if os.path.exists(hist_path):
        tmp = pd.read_csv(hist_path, dtype=str).fillna("")
        _ensure_cols(tmp, {"data","employee_id","turno"}, "history.csv")
        tmp["data"] = tmp["data"].astype(str).str.strip()
        tmp = attach_calendar(tmp, calendar_df)
        history_df = tmp

    availability_df = load_availability(os.path.join(data_dir, "availability.csv"), employees_df, calendar_df)

    return LoadedData(
        cfg=cfg,
        calendar_df=calendar_df,
        employees_df=employees_df,
        shifts_df=shifts_df,
        month_plan_df=month_plan_df,
        groups_total_expanded=groups_total_expanded,
        groups_role_min_expanded=groups_role_min_expanded,
        history_df=history_df,
        availability_df=availability_df,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Loader clinica — Step A (v2)")
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
        print(f"Esportati CSV di debug in: {outdir}")
