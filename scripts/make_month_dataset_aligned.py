import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ModuleNotFoundError:  # pragma: no cover - fallback for environments senza Faker
    class _FallbackFaker:
        _NAMES = [
            "Mario Rossi",
            "Giulia Bianchi",
            "Luca Conti",
            "Sara Greco",
            "Paolo Ferri",
            "Anna Romano",
        ]

        def name(self) -> str:
            return random.choice(self._NAMES)

    class Faker:  # type: ignore[misc]
        def __init__(self, *_args, **_kwargs) -> None:
            self._impl = _FallbackFaker()

        def name(self) -> str:  # pragma: no cover - delega al fallback
            return self._impl.name()


SEED = 42
MONTH = "2025-11"
OUT_DIR = Path("data_fake_month")
TARGET_SHIFTS_PER_EMP = 20

random.seed(SEED)
np.random.seed(SEED)
fake = Faker("it_IT")


@dataclass(frozen=True)
class RoleSpec:
    pool_id: str
    min_staff: int
    max_staff: int
    hours_options: tuple[int, ...]
    can_work_night: bool = False
    max_nights_week: int = 0
    max_nights_month: int = 0


SHIFT_DEFINITIONS = {
    "M": {
        "nome": "Mattino",
        "start": "07:00",
        "end": "14:30",
        "break_min": 30,
        "duration_min": 420,
        "crosses_midnight": 0,
    },
    "P": {
        "nome": "Pomeriggio",
        "start": "14:00",
        "end": "21:30",
        "break_min": 30,
        "duration_min": 420,
        "crosses_midnight": 0,
    },
    "N": {
        "nome": "Notte",
        "start": "21:00",
        "end": "07:30",
        "break_min": 30,
        "duration_min": 630,
        "crosses_midnight": 1,
    },
    "R": {
        "nome": "Riposo",
        "start": "",
        "end": "",
        "break_min": 0,
        "duration_min": 0,
        "crosses_midnight": 0,
    },
    "F": {
        "nome": "Ferie",
        "start": "",
        "end": "",
        "break_min": 0,
        "duration_min": 0,
        "crosses_midnight": 0,
    },
}

SHIFT_ROLE_ELIGIBILITY = {
    "M": {
        "infermiere": True,
        "oss": True,
        "medico": True,
        "amministrativo": True,
    },
    "P": {
        "infermiere": True,
        "oss": True,
        "medico": True,
        "amministrativo": False,
    },
    "N": {
        "infermiere": True,
        "oss": False,
        "medico": False,
        "amministrativo": False,
    },
    "R": {
        "infermiere": True,
        "oss": True,
        "medico": True,
        "amministrativo": True,
    },
    "F": {
        "infermiere": True,
        "oss": True,
        "medico": True,
        "amministrativo": True,
    },
}

DEPARTMENTS = {
    "degenza": {
        "label": "Degenza",
        "coverage_codes": {"M": "DEG_DAY_M", "P": "DEG_DAY_P"},
        "shifts": ("M", "P"),
        "roles": {
            "infermiere": RoleSpec("DN_NURSE", 12, 12, (88,), True, 2, 8),
            "oss": RoleSpec("DN_OSS", 9, 9, (70,), False, 0, 0),
            "medico": RoleSpec("DN_MED", 5, 5, (84,), False, 0, 0),
        },
        "coverage_ratios": {
            "M": {"infermiere": 0.20, "oss": 0.12, "medico": 0.10},
            "P": {"infermiere": 0.16, "oss": 0.10, "medico": 0.10},
        },
        "shift_map": {
            "M": {"enabled": True, "start_override": "06:30", "end_override": ""},
            "P": {"enabled": True, "start_override": "", "end_override": ""},
        },
    },
    "pronto_soccorso": {
        "label": "Pronto soccorso",
        "coverage_codes": {"M": "PS_DAY_M", "P": "PS_DAY_P", "N": "PS_NIGHT_N"},
        "shifts": ("M", "P", "N"),
        "roles": {
            "infermiere": RoleSpec("PS_NURSE", 17, 17, (105,), True, 3, 10),
            "oss": RoleSpec("PS_OSS", 9, 9, (56,), False, 0, 0),
            "medico": RoleSpec("PS_MED", 6, 6, (70,), False, 0, 0),
        },
        "coverage_ratios": {
            "M": {"infermiere": 0.15, "oss": 0.10, "medico": 0.09},
            "P": {"infermiere": 0.13, "oss": 0.10, "medico": 0.09},
            "N": {"infermiere": 0.05},
        },
        "shift_map": {
            "M": {"enabled": True, "start_override": "07:00", "end_override": ""},
            "P": {"enabled": True, "start_override": "", "end_override": ""},
            "N": {"enabled": True, "start_override": "", "end_override": "07:30"},
        },
    },
    "ambulatorio": {
        "label": "Ambulatorio",
        "coverage_codes": {"M": "AMB_DAY_M"},
        "shifts": ("M",),
        "roles": {
            "infermiere": RoleSpec("AMB_NURSE", 4, 4, (56,)),
            "medico": RoleSpec("AMB_MED", 3, 3, (70,)),
            "amministrativo": RoleSpec("AMB_AMM", 5, 5, (42,)),
        },
        "coverage_ratios": {
            "M": {
                "infermiere": 0.25,
                "medico": 0.30,
                "amministrativo": 0.20,
            },
        },
        "shift_map": {
            "M": {
                "enabled": True,
                "start_override": "07:30",
                "end_override": "13:30",
            },
            "P": {"enabled": False, "start_override": "", "end_override": ""},
        },
    },
}

HOLIDAYS = {
    f"{MONTH}-01": "Ognissanti",
    f"{MONTH}-04": "FestivitÃ  locale",
}


def make_days(month: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(f"{month}-01")
    end = start + pd.offsets.MonthEnd(0)
    return pd.date_range(start, end, freq="D")


def generate_employees() -> pd.DataFrame:
    rows = []
    emp_idx = 1
    for reparto_id, dept_cfg in DEPARTMENTS.items():
        for role, spec in dept_cfg["roles"].items():
            n_staff = random.randint(spec.min_staff, spec.max_staff)
            for _ in range(n_staff):
                employee_id = f"E{emp_idx:03d}"
                emp_idx += 1
                rows.append(
                    {
                        "employee_id": employee_id,
                        "nome": fake.name(),
                        "role": role,
                        "ruolo": role,
                        "reparto_id": reparto_id,
                        "reparto_label": dept_cfg["label"],
                        "ore_dovute_mese_h": random.choice(spec.hours_options),
                        "saldo_prog_iniziale_h": random.choice([-4, -2, 0, 2, 4]),
                        "max_balance_delta_month_h": 20,
                        "max_month_hours_h": 200,
                        "max_week_hours_h": 50,
                        "can_work_night": "yes" if spec.can_work_night else "no",
                        "max_nights_week": spec.max_nights_week if spec.can_work_night else 0,
                        "max_nights_month": spec.max_nights_month if spec.can_work_night else 0,
                        "saturday_count_ytd": random.randint(0, 4),
                        "sunday_count_ytd": random.randint(0, 4),
                        "holiday_count_ytd": random.randint(0, 3),
                        "pool_id": spec.pool_id,
                        "cross_max_shifts_month": "",
                    }
                )
    columns = [
        "employee_id",
        "nome",
        "role",
        "ruolo",
        "reparto_id",
        "reparto_label",
        "ore_dovute_mese_h",
        "saldo_prog_iniziale_h",
        "max_balance_delta_month_h",
        "max_month_hours_h",
        "max_week_hours_h",
        "can_work_night",
        "max_nights_week",
        "max_nights_month",
        "saturday_count_ytd",
        "sunday_count_ytd",
        "holiday_count_ytd",
        "pool_id",
        "cross_max_shifts_month",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_month_plan(days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        for reparto_id, dept_cfg in DEPARTMENTS.items():
            for shift in dept_cfg["shifts"]:
                coverage_code = dept_cfg["coverage_codes"][shift]
                rows.append(
                    {
                        "data": day_str,
                        "reparto_id": reparto_id,
                        "shift_code": shift,
                        "coverage_code": coverage_code,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["data", "reparto_id", "shift_code", "coverage_code"]).reset_index(
        drop=True
    )


def compute_coverage_requirements(
    employees_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_counts = (
        employees_df.groupby(["reparto_id", "role"])["employee_id"].count().rename("staff")
    )
    role_counts = role_counts.reset_index()

    cov_rows = []
    group_rows = []
    for reparto_id, dept_cfg in DEPARTMENTS.items():
        dept_counts = role_counts.loc[role_counts["reparto_id"] == reparto_id]
        count_lookup = {row.role: int(row.staff) for row in dept_counts.itertuples(index=False)}
        for shift_code, ratios in dept_cfg["coverage_ratios"].items():
            coverage_code = dept_cfg["coverage_codes"][shift_code]
            for role, ratio in ratios.items():
                staff_available = count_lookup.get(role, 0)
                if staff_available <= 0:
                    continue
                min_staff = max(1, math.ceil(staff_available * ratio))
                cov_rows.append(
                    {
                        "coverage_code": coverage_code,
                        "shift_code": shift_code,
                        "reparto_id": reparto_id,
                        "gruppo": role,
                        "role": role,
                        "ruolo": role,
                        "min_ruolo": int(min_staff),
                    }
                )
                group_rows.append(
                    {
                        "coverage_code": coverage_code,
                        "shift_code": shift_code,
                        "reparto_id": reparto_id,
                        "gruppo": role,
                        "total_staff": int(min_staff),
                        "ruoli_totale": role,
                        "overstaff_cap": "",
                    }
                )
    cov_df = pd.DataFrame(
        cov_rows,
        columns=[
            "coverage_code",
            "shift_code",
            "reparto_id",
            "gruppo",
            "role",
            "ruolo",
            "min_ruolo",
        ],
    )
    groups_df = pd.DataFrame(
        group_rows,
        columns=[
            "coverage_code",
            "shift_code",
            "reparto_id",
            "gruppo",
            "total_staff",
            "ruoli_totale",
            "overstaff_cap",
        ],
    )
    return cov_df, groups_df


def generate_history(days: pd.DatetimeIndex, employees_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["data", "employee_id", "turno"])


def generate_availability(days: pd.DatetimeIndex, employees_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["data", "employee_id", "turno"])


def generate_leaves(days: pd.DatetimeIndex, employees_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["employee_id", "date_from", "date_to", "type", "is_planned"])


def generate_locks(
    month_plan_df: pd.DataFrame,
    employees_df: pd.DataFrame,
) -> pd.DataFrame:
    if month_plan_df.empty:
        return pd.DataFrame(columns=["employee_id", "slot_id", "lock", "note"])

    slots_df = month_plan_df.copy()
    slots_df = slots_df.sort_values(["data", "reparto_id", "shift_code", "coverage_code"]).reset_index(
        drop=True
    )
    slots_df.insert(0, "slot_id", range(1, len(slots_df) + 1))

    slots_by_reparto = {
        reparto: group for reparto, group in slots_df.groupby("reparto_id")
    }

    n_locks = max(1, int(len(employees_df) * 0.03))
    selected_emps = employees_df.sample(n_locks, random_state=SEED + 1)

    lock_rows = []
    used_slots: set[int] = set()
    for row in selected_emps.itertuples(index=False):
        reparto_slots = slots_by_reparto[row.reparto_id]
        role_spec = DEPARTMENTS[row.reparto_id]["roles"][row.role]
        if not role_spec.can_work_night:
            eligible_slots = reparto_slots[reparto_slots["shift_code"] != "N"]
            if eligible_slots.empty:
                eligible_slots = reparto_slots
        else:
            eligible_slots = reparto_slots
        # Evita di assegnare più lock allo stesso slot.
        eligible_slots = eligible_slots[~eligible_slots["slot_id"].isin(used_slots)]
        if eligible_slots.empty:
            continue
        slot_row = eligible_slots.sample(1, random_state=random.randint(0, 1_000_000)).iloc[0]
        used_slots.add(int(slot_row.slot_id))
        lock_rows.append(
            {
                "employee_id": row.employee_id,
                "slot_id": int(slot_row.slot_id),
                "lock": -1,
                "note": "",
            }
        )

    return pd.DataFrame(lock_rows, columns=["employee_id", "slot_id", "lock", "note"])


def generate_role_pools() -> pd.DataFrame:
    rows = []
    for reparto_id, dept_cfg in DEPARTMENTS.items():
        for role, spec in dept_cfg["roles"].items():
            rows.append(
                {
                    "role": role,
                    "pool_id": spec.pool_id,
                    "reparto_id": reparto_id,
                }
            )
    return pd.DataFrame(rows, columns=["role", "pool_id", "reparto_id"])


def generate_shifts() -> pd.DataFrame:
    rows = []
    for shift_id, cfg in SHIFT_DEFINITIONS.items():
        rows.append(
            {
                "shift_id": shift_id,
                "nome": cfg["nome"],
                "start": cfg["start"],
                "end": cfg["end"],
                "break_min": cfg["break_min"],
                "duration_min": cfg["duration_min"],
                "crosses_midnight": cfg["crosses_midnight"],
            }
        )
    columns = [
        "shift_id",
        "nome",
        "start",
        "end",
        "break_min",
        "duration_min",
        "crosses_midnight",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("shift_id").reset_index(drop=True)


def generate_reparto_shift_map() -> pd.DataFrame:
    rows = []
    for reparto_id, dept_cfg in DEPARTMENTS.items():
        for shift_code, cfg in dept_cfg["shift_map"].items():
            rows.append(
                {
                    "reparto_id": reparto_id,
                    "shift_code": shift_code,
                    "enabled": str(cfg["enabled"]).lower(),
                    "start_override": cfg["start_override"],
                    "end_override": cfg["end_override"],
                }
            )
    columns = ["reparto_id", "shift_code", "enabled", "start_override", "end_override"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["reparto_id", "shift_code"]).reset_index(drop=True)


def generate_shift_role_eligibility() -> pd.DataFrame:
    rows = []
    for shift_code, roles in SHIFT_ROLE_ELIGIBILITY.items():
        for role, allowed in roles.items():
            rows.append(
                {
                    "shift_code": shift_code,
                    "role": role.upper(),
                    "allowed": str(bool(allowed)),
                }
            )
    return pd.DataFrame(rows, columns=["shift_code", "role", "allowed"]).sort_values(
        ["shift_code", "role"]
    )


def generate_holidays() -> pd.DataFrame:
    return pd.DataFrame(
        (
            {"data": date_str, "descrizione": descr}
            for date_str, descr in HOLIDAYS.items()
        ),
        columns=["data", "descrizione"],
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    days = make_days(MONTH)
    employees_df = generate_employees()
    employees_df.to_csv(OUT_DIR / "employees.csv", index=False)

    month_plan_df = build_month_plan(days)
    month_plan_df.to_csv(OUT_DIR / "month_plan.csv", index=False)

    coverage_roles_df, coverage_groups_df = compute_coverage_requirements(employees_df)
    coverage_roles_df.to_csv(OUT_DIR / "coverage_roles.csv", index=False)
    coverage_groups_df.to_csv(OUT_DIR / "coverage_groups.csv", index=False)

    shifts_df = generate_shifts()
    shifts_df.to_csv(OUT_DIR / "shifts.csv", index=False)

    reparto_shift_map_df = generate_reparto_shift_map()
    reparto_shift_map_df.to_csv(OUT_DIR / "reparto_shift_map.csv", index=False)

    shift_role_eligibility_df = generate_shift_role_eligibility()
    shift_role_eligibility_df.to_csv(OUT_DIR / "shift_role_eligibility.csv", index=False)

    history_df = generate_history(days, employees_df)
    history_path = OUT_DIR / "history.csv"
    if history_df.empty:
        history_path.unlink(missing_ok=True)
    else:
        history_df.to_csv(history_path, index=False)

    availability_df = generate_availability(days, employees_df)
    availability_path = OUT_DIR / "availability.csv"
    if availability_df.empty:
        availability_path.unlink(missing_ok=True)
    else:
        availability_df.to_csv(availability_path, index=False)

    leaves_df = generate_leaves(days, employees_df)
    leaves_path = OUT_DIR / "leaves.csv"
    if leaves_df.empty:
        leaves_path.unlink(missing_ok=True)
    else:
        leaves_df.to_csv(leaves_path, index=False)

    locks_df = generate_locks(month_plan_df, employees_df)
    locks_df.to_csv(OUT_DIR / "locks.csv", index=False)

    role_dept_pools_df = generate_role_pools()
    role_dept_pools_df.to_csv(OUT_DIR / "role_dept_pools.csv", index=False)

    holidays_df = generate_holidays()
    holidays_df.to_csv(OUT_DIR / "holidays.csv", index=False)

    req_summary = (
        coverage_roles_df.groupby(["coverage_code", "shift_code", "reparto_id", "role"])[
            "min_ruolo"
        ]
        .sum()
        .reset_index()
    )
    daily_required = int(req_summary["min_ruolo"].sum())
    monthly_required = daily_required * len(days)
    estimated_capacity = int(len(employees_df) * TARGET_SHIFTS_PER_EMP)
    utilization = monthly_required / max(estimated_capacity, 1)

    print("====== REPORT ======")
    print(f"Dipendenti: {len(employees_df)} | Giorni: {len(days)}")
    print(f"Turni richiesti/giorno (somma min_ruolo): {daily_required}")
    print(f"Turni richiesti nel mese: {monthly_required}")
    print(f"Capacita stimata (turni): {estimated_capacity}")
    print(f"Utilization stimata: {utilization:.1%} (target ~50-85%)")
    print("====================")
    print(f"Dataset generato in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
