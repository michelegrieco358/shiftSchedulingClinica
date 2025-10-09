from __future__ import annotations

import pandas as pd


def _normalize_roles(series: pd.Series) -> pd.Series:
    cleaned = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return cleaned[(cleaned != "") & (cleaned != "NAN")]


def _build_absence_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if "employee_id" not in df.columns:
        return None

    work = df.copy()
    date_col = next((col for col in ("slot_date", "date", "data") if col in work.columns), None)
    if date_col is None:
        return None

    work["employee_id"] = work["employee_id"].astype(str).str.strip()
    work["slot_date"] = pd.to_datetime(work[date_col]).dt.date

    mask = pd.Series(True, index=work.index)
    if "kind" in work.columns:
        kind_series = work["kind"].astype(str).str.strip().str.lower()
        mask &= kind_series.isin({"full_day", "full-day", "full"})
    elif "tipo" in work.columns:
        tipo_series = work["tipo"].astype(str).str.strip().str.lower()
        mask &= tipo_series.isin({"full_day", "full-day"})
    elif "is_absent" in work.columns:
        abs_series = work["is_absent"].astype(str).str.strip().str.lower()
        mask &= abs_series.isin({"1", "true", "t", "yes", "y", "si"})

    filtered = work.loc[mask, ["employee_id", "slot_date"]].drop_duplicates()
    if filtered.empty:
        return None
    return filtered.reset_index(drop=True)

def build_all(dfs: dict, cfg: dict) -> dict:
    """Crea dizionari e strutture dati derivate dai DataFrame principali."""
    bundle = {}

    # Estraggo i DataFrame principali
    df_employees = dfs.get("employees")
    df_slots = dfs.get("shift_slots")
    df_elig = dfs.get("shift_role_eligibility")
    df_pools = dfs.get("role_dept_pools")
    df_pre = dfs.get("preassignments")
    df_abs = (
        dfs.get("absences")
        or dfs.get("leaves_days_df")
        or dfs.get("leaves_df")
    )
    df_availability = dfs.get("availability_df") or dfs.get("availability")

    if df_employees is None or df_slots is None:
        raise ValueError("Mancano DataFrame essenziali: employees o shift_slots")
    if "is_night" not in df_slots.columns:
        df_slots = df_slots.copy()
        df_slots["is_night"] = False
    if "can_work_night" not in df_employees.columns:
        df_employees = df_employees.copy()
        df_employees["can_work_night"] = True
    
    # (1) Indici contigui -----------------------------------------------
    # Dipendenti
    employee_ids = sorted(df_employees["employee_id"].unique())
    eid_of = {eid: i for i, eid in enumerate(employee_ids)}
    emp_of = {i: eid for eid, i in eid_of.items()}
    df_employees["employee_id2"] = df_employees["employee_id"].map(eid_of)

    # Slot
    slot_ids = sorted(df_slots["slot_id"].unique())
    sid_of = {sid: i for i, sid in enumerate(slot_ids)}
    slot_of = {i: sid for sid, i in sid_of.items()}
    df_slots["slot_id2"] = df_slots["slot_id"].map(sid_of)

    # Giorni (dal calendario esteso, non dagli slot)
    df_cal = dfs.get("calendar_df") 
    if df_cal is None:
        raise ValueError("Manca il DataFrame di calendario (calendar_df).")

    day_values = sorted(pd.to_datetime(df_cal["data"]).dt.date.unique())

    did_of = {d: i for i, d in enumerate(day_values)}
    date_of = {i: d for d, i in did_of.items()}

    # Aggiungi al bundle
    bundle.update({
        "eid_of": eid_of,
        "emp_of": emp_of,
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": len(eid_of),
        "num_slots": len(sid_of),
        "num_days": len(did_of),
    })

    # (2) Mappe slot (date, reparti, turni, ecc.)

    # Mappe slot: date, reparto, shift, durata (dopo override già nel DF)
    slot_date = df_slots.set_index("slot_id")["date"].to_dict()
    slot_reparto = df_slots.set_index("slot_id")["reparto_id"].to_dict()
    slot_shiftcode = df_slots.set_index("slot_id")["shift_code"].to_dict()
    slot_duration_min = df_slots.set_index("slot_id")["duration_min"].astype(int).to_dict()

    # Versione indicizzata (usa slot_id2 -> day_id)
    slot_date2 = {sid_of[sid]: did_of[pd.to_datetime(dt).date()]
                for sid, dt in slot_date.items()}

    bundle.update({
        "slot_date": slot_date,
        "slot_reparto": slot_reparto,
        "slot_shiftcode": slot_shiftcode,
        "slot_duration_min": slot_duration_min,
        "slot_date2": slot_date2,
    })

    # Giorni toccati da ciascuno slot (utile per notti/SN)
    slot_days_touched = {}
    for _, row in df_slots.iterrows():
        sid = row["slot_id"]
        start = pd.to_datetime(row["start_datetime"])
        end = pd.to_datetime(row["end_datetime"])
        days = pd.date_range(start.normalize(), end.normalize()).date
        slot_days_touched[sid] = [did_of[d] for d in days if d in did_of]

    bundle["slot_days_touched"] = slot_days_touched


    # (3) Idoneita e coverage

    role_column = "role" if "role" in df_employees.columns else "ruolo"
    if role_column not in df_employees.columns:
        raise ValueError("preprocessing: colonna ruolo mancante in df_employees")

    slot_base = df_slots.loc[:, ["slot_id", "slot_id2", "reparto_id", "shift_code", "date", "is_night"]].copy()
    slot_base = slot_base.rename(columns={"reparto_id": "slot_reparto_id"})
    slot_base["slot_reparto_id"] = (
        slot_base["slot_reparto_id"].astype(str).str.strip().str.upper()
    )
    slot_base["shift_code"] = slot_base["shift_code"].astype(str).str.strip().str.upper()
    slot_base["slot_date"] = pd.to_datetime(slot_base["date"]).dt.date
    slot_base["is_night"] = slot_base["is_night"].fillna(False).astype(bool)

    if df_elig is not None and not df_elig.empty:
        allowed_roles = (
            df_elig.loc[df_elig["allowed"] == True, ["shift_code", "role"]]
            .assign(
                shift_code=lambda df: df["shift_code"].astype(str).str.strip().str.upper(),
                role=lambda df: df["role"].astype(str).str.strip().str.upper(),
            )
            .drop_duplicates()
        )
        slot_role = slot_base.merge(allowed_roles, on="shift_code", how="inner")
    else:
        roles = _normalize_roles(df_employees[role_column])
        slot_role = slot_base.assign(key=1).merge(
            pd.DataFrame({"role": roles.unique(), "key": 1}),
            on="key",
            how="inner",
        ).drop(columns="key")

    slot_role["role"] = slot_role["role"].astype(str).str.strip().str.upper()

    emp_base = df_employees.loc[:, ["employee_id", "employee_id2", role_column, "reparto_id", "pool_id"]].copy()
    emp_base = emp_base.rename(columns={role_column: "role", "reparto_id": "employee_reparto_id"})
    emp_base["role"] = emp_base["role"].astype(str).str.strip().str.upper()
    emp_base["employee_reparto_id"] = (
        emp_base["employee_reparto_id"].astype(str).str.strip().str.upper()
    )
    emp_base["pool_id"] = emp_base["pool_id"].fillna("").astype(str).str.strip()

    # 1) dipendenti del reparto dello slot
    in_reparto_candidates = slot_role.merge(
        emp_base,
        left_on=["role", "slot_reparto_id"],
        right_on=["role", "employee_reparto_id"],
        how="inner",
    )

    # 2) dipendenti abilitati tramite pool
    cross_candidates = pd.DataFrame(columns=in_reparto_candidates.columns)
    if df_pools is not None and not df_pools.empty:
        pool_tbl = df_pools.loc[:, ["role", "pool_id", "reparto_id"]].copy()
        pool_tbl["role"] = pool_tbl["role"].astype(str).str.strip().str.upper()
        pool_tbl["pool_id"] = pool_tbl["pool_id"].fillna("").astype(str).str.strip()
        pool_tbl["reparto_id"] = pool_tbl["reparto_id"].astype(str).str.strip().str.upper()
        pool_tbl = pool_tbl.drop_duplicates()

        cross_slot = slot_role.merge(
            pool_tbl,
            left_on=["role", "slot_reparto_id"],
            right_on=["role", "reparto_id"],
            how="inner",
        ).drop(columns=["reparto_id"])

        cross_candidates = cross_slot.merge(
            emp_base.loc[emp_base["pool_id"] != ""],
            on=["role", "pool_id"],
            how="inner",
        )
        cross_candidates = cross_candidates[
            cross_candidates["slot_reparto_id"] != cross_candidates["employee_reparto_id"]
        ]

    candidates = pd.concat([in_reparto_candidates, cross_candidates], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["employee_id2", "slot_id2"])

    if df_pre is not None and not df_pre.empty:
        forbidden_tbl = df_pre.loc[df_pre["lock"].astype(int) == -1, ["employee_id", "slot_id"]].drop_duplicates()
        candidates = candidates.merge(
            forbidden_tbl,
            on=["employee_id", "slot_id"],
            how="left",
            indicator="_forbid",
        )
        candidates = candidates[candidates["_forbid"] != "both"].drop(columns="_forbid")

    absence_tbl = _build_absence_index(df_abs)
    if absence_tbl is not None and not absence_tbl.empty:
        candidates = candidates.merge(
            absence_tbl,
            on=["employee_id", "slot_date"],
            how="left",
            indicator="_abs",
        )
        candidates = candidates[candidates["_abs"] != "both"].drop(columns="_abs")

    if df_availability is not None and not df_availability.empty:
        availability_tbl = (
            df_availability.loc[:, ["employee_id", "turno", "data"]]
            .assign(
                employee_id=lambda df: df["employee_id"].astype(str).str.strip(),
                turno=lambda df: df["turno"].astype(str).str.strip().str.upper(),
                slot_date=lambda df: pd.to_datetime(df["data"]).dt.date,
            )
            .drop(columns="data")
            .drop_duplicates()
        )
        candidates = candidates.merge(
            availability_tbl,
            left_on=["employee_id", "slot_date", "shift_code"],
            right_on=["employee_id", "slot_date", "turno"],
            how="left",
            indicator="_availability",
        )
        candidates = candidates[candidates["_availability"] != "both"]
        drop_cols = [col for col in ("_availability", "turno") if col in candidates.columns]
        if drop_cols:
            candidates = candidates.drop(columns=drop_cols)

    if "is_night" in candidates.columns:
        night_mask = candidates["is_night"].fillna(False)
        if "can_work_night" in candidates.columns:
            can_night = candidates["can_work_night"].fillna(True).astype(bool)
            candidates = candidates[~(night_mask & ~can_night)]
        else:
            candidates = candidates[~night_mask]

    eligible_sids = {eid_of[eid]: [] for eid in df_employees["employee_id"].unique()}
    eligible_eids = {sid_of[sid]: [] for sid in df_slots["slot_id"].unique()}

    if not candidates.empty:
        for e2, slot_ids in candidates.groupby("employee_id2")["slot_id2"]:
            eligible_sids[e2] = slot_ids.tolist()
        for sid2, employee_ids in candidates.groupby("slot_id2")["employee_id2"]:
            eligible_eids[sid2] = employee_ids.tolist()

    bundle.update(
        {
            "eligible_sids": eligible_sids,
            "eligible_eids": eligible_eids,
        }
    )
    # (4) Altri set utili (assenze, festivi, ecc.)

    return bundle
