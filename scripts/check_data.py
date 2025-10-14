"""Quick data consistency checks without pandas."""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class DatasetSummary:
    label: str
    rows: int


class ValidationError(Exception):
    pass


def _read_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"File CSV non trovato: {path}")

    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except UnicodeDecodeError:
        handle = path.open("r", encoding="utf-8", newline="")
    with handle as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValidationError(f"{path.name}: nessuna intestazione trovata")
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {k: (row.get(k, "") or "").strip() for k in reader.fieldnames}
            rows.append(cleaned)
    return rows, reader.fieldnames


def _require_columns(columns: Sequence[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValidationError(f"{label}: colonne mancanti {missing}")


def _parse_date(value: str, label: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValidationError(f"{label}: data non valida '{value}': {exc}") from exc


def _parse_int(value: str, label: str, allow_negative: bool = False) -> int:
    try:
        num = int(float(value)) if value != "" else 0
    except ValueError as exc:
        raise ValidationError(f"{label}: valore non numerico '{value}'") from exc
    if not allow_negative and num < 0:
        raise ValidationError(f"{label}: valore negativo non ammesso ({value})")
    return num


def _parse_float(value: str, label: str, allow_negative: bool = False) -> float:
    try:
        num = float(value)
    except ValueError as exc:
        raise ValidationError(f"{label}: valore non numerico '{value}'") from exc
    if not allow_negative and num < 0:
        raise ValidationError(f"{label}: valore negativo non ammesso ({value})")
    return num


def _count_weeks_in_horizon(start: date, end: date) -> int:
    if end < start:
        return 0
    seen = set()
    current = start
    while current <= end:
        week_start = current - timedelta(days=current.isoweekday() - 1)
        seen.add(week_start)
        current += timedelta(days=1)
    return len(seen)


def _resolve_allowed_roles(defaults: dict, fallback_roles: Iterable[str]) -> list[str]:
    allowed_cfg = defaults.get("allowed_roles")
    roles: list[str] = []
    if isinstance(allowed_cfg, str):
        roles = [
            part.strip()
            for part in allowed_cfg.replace(",", "|").split("|")
            if part.strip()
        ]
    elif isinstance(allowed_cfg, (list, tuple, set)):
        roles = [str(part).strip() for part in allowed_cfg if str(part).strip()]
    if not roles:
        roles = [str(part).strip() for part in fallback_roles if str(part).strip()]
    deduped: list[str] = []
    seen = set()
    for role in roles:
        if role not in seen:
            deduped.append(role)
            seen.add(role)
    return deduped


def _resolve_allowed_departments(defaults: dict) -> list[str]:
    departments_cfg = defaults.get("departments")
    departments: list[str] = []
    if isinstance(departments_cfg, str):
        departments = [
            part.strip()
            for part in departments_cfg.replace(",", "|").split("|")
            if part.strip()
        ]
    elif isinstance(departments_cfg, (list, tuple, set)):
        departments = [str(part).strip() for part in departments_cfg if str(part).strip()]
    if not departments:
        raise ValidationError(
            "config: defaults.departments deve essere una lista non vuota di reparti ammessi"
        )
    deduped: list[str] = []
    seen = set()
    for dept in departments:
        if dept not in seen:
            deduped.append(dept)
            seen.add(dept)
    return deduped


@dataclass
class ValidationResult:
    summaries: list[DatasetSummary]
    horizon_start: date
    horizon_end: date


def _check_employees(
    path: Path,
    defaults: dict,
    role_defaults: dict,
    weeks_in_horizon: int,
    horizon_days: int,
) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    base_required = {
        "employee_id",
        "nome",
        "ruolo",
        "ore_dovute_mese_h",
        "saldo_prog_iniziale_h",
    }
    if "reparto" in columns:
        dept_col = "reparto"
    elif "reparto_id" in columns:
        dept_col = "reparto_id"
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti {{'reparto' o 'reparto_id'}}"
        )
    _require_columns(columns, base_required | {dept_col}, path.name)

    ids = [row["employee_id"] for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValidationError(
            f"{path.name}: employee_id duplicati trovati: {sorted(duplicates)}"
        )

    allowed_roles = _resolve_allowed_roles(defaults, {row["ruolo"] for row in rows})
    allowed_departments = _resolve_allowed_departments(defaults)

    def _coerce_bool(value, label: str, allow_empty: bool = False) -> bool | None:
        if value is None:
            if allow_empty:
                return None
            raise ValidationError(f"{label} mancante")
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "":
                if allow_empty:
                    return None
                raise ValidationError(f"{label} mancante")
            if s in {"true", "1", "yes", "y", "si", "s"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
        elif isinstance(value, (int, float)):
            if value in {0, 1}:
                return bool(int(value))
        elif isinstance(value, bool):
            return value
        raise ValidationError(f"{label}: valore booleano non riconosciuto ({value!r})")

    def _coerce_nonneg_int(
        value, label: str, allow_empty: bool = False
    ) -> int | None:
        if value is None:
            if allow_empty:
                return None
            raise ValidationError(f"{label} mancante")
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                if allow_empty:
                    return None
                raise ValidationError(f"{label} mancante")
            try:
                num = float(s)
            except ValueError as exc:
                raise ValidationError(f"{label}: valore non numerico ({value!r})") from exc
        else:
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{label}: valore non numerico ({value!r})") from exc
        if num < 0:
            raise ValidationError(f"{label}: valore negativo non ammesso ({num})")
        return int(round(num))

    night_defaults = defaults.get("night", {}) or {}
    global_can_work = _coerce_bool(
        night_defaults.get("can_work_night"),
        "config: defaults.night.can_work_night",
    )
    global_max_week = _coerce_nonneg_int(
        night_defaults.get("max_per_week"),
        "config: defaults.night.max_per_week",
    )
    global_max_month = _coerce_nonneg_int(
        night_defaults.get("max_per_month"),
        "config: defaults.night.max_per_month",
    )

    for row in rows:
        role = row["ruolo"].strip()
        if role not in allowed_roles:
            raise ValidationError(
                f"{path.name}: ruolo non ammesso '{role}'. Ruoli ammessi: {allowed_roles}"
            )
        dept = row.get(dept_col, "").strip()
        if dept_col != "reparto":
            row["reparto"] = dept
        if not dept:
            raise ValidationError(
                f"{path.name}: reparto vuoto per employee_id {row['employee_id']}"
            )
        if dept not in allowed_departments:
            raise ValidationError(
                f"{path.name}: reparto '{dept}' non previsto in defaults.departments"
            )

        ore = row.get("ore_dovute_mese_h", "")
        saldo = row.get("saldo_prog_iniziale_h", "")
        contract_hours = _parse_float(ore, f"{path.name}: ore_dovute_mese_h")
        _parse_float(saldo, f"{path.name}: saldo_prog_iniziale_h", allow_negative=True)

        month_cap = row.get("max_month_hours_h", "").strip()
        if month_cap:
            month_hours = _parse_float(month_cap, f"{path.name}: max_month_hours_h")
            if month_hours + 1e-9 < contract_hours:
                raise ValidationError(
                    f"{path.name}: max_month_hours_h deve essere ≥ ore_dovute_mese_h per employee_id {row['employee_id']}"
                )

        week_cap = row.get("max_week_hours_h", "").strip()
        if week_cap:
            if weeks_in_horizon <= 0:
                raise ValidationError("config: orizzonte senza settimane per validare max_week_hours_h")
            if horizon_days <= 0:
                raise ValidationError("config: orizzonte senza giorni per validare max_week_hours_h")
            week_hours = _parse_float(week_cap, f"{path.name}: max_week_hours_h")
            # Stessa formula del loader: distribuzione delle ore contrattuali su
            # una settimana "media" del mese (ore_mese / giorni_orizzonte * 7)
            # così da applicare un cap uniforme anche alle settimane parziali.
            weekly_theoretical = (
                contract_hours / horizon_days * 7.0 if horizon_days else 0.0
            )
            if week_hours + 1e-9 < weekly_theoretical:
                raise ValidationError(
                    f"{path.name}: max_week_hours_h inferiore alle ore teoriche settimanali per employee_id {row['employee_id']}"
                )

        role_cfg = role_defaults.get(role, {}) or {}
        role_can = _coerce_bool(
            role_cfg.get("can_work_night"),
            f"config: roles.{role}.can_work_night",
            allow_empty=True,
        )
        if role_can is None:
            role_can = global_can_work

        role_night_cfg = role_cfg.get("night", {}) or {}
        role_week = _coerce_nonneg_int(
            role_night_cfg.get("max_per_week"),
            f"config: roles.{role}.night.max_per_week",
            allow_empty=True,
        )
        if role_week is None:
            role_week = global_max_week
        role_month = _coerce_nonneg_int(
            role_night_cfg.get("max_per_month"),
            f"config: roles.{role}.night.max_per_month",
            allow_empty=True,
        )
        if role_month is None:
            role_month = global_max_month

        emp_can = _coerce_bool(
            row.get("can_work_night", ""),
            f"{path.name}: can_work_night",
            allow_empty=True,
        )
        if emp_can is None:
            emp_can = role_can

        emp_week = _coerce_nonneg_int(
            row.get("max_nights_week", ""),
            f"{path.name}: max_nights_week",
            allow_empty=True,
        )
        if emp_week is None:
            emp_week = role_week

        emp_month = _coerce_nonneg_int(
            row.get("max_nights_month", ""),
            f"{path.name}: max_nights_month",
            allow_empty=True,
        )
        if emp_month is None:
            emp_month = role_month

        for ytd_col in ("saturday_count_ytd", "sunday_count_ytd", "holiday_count_ytd"):
            val = row.get(ytd_col, "0") or "0"
            _parse_int(val, f"{path.name}: {ytd_col}")

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_shifts(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(
        columns,
        {"shift_id", "start", "end", "break_min", "duration_min", "crosses_midnight"},
        path.name,
    )

    seen = {}
    for row in rows:
        sid = row["shift_id"]
        if sid in seen:
            prev = seen[sid]
            if (
                prev["start"] != row["start"]
                or prev["end"] != row["end"]
                or prev["duration_min"] != row["duration_min"]
                or prev["crosses_midnight"] != row["crosses_midnight"]
            ):
                raise ValidationError(
                    f"{path.name}: definizioni multiple incoerenti per shift_id '{sid}'"
                )
        else:
            seen[sid] = row

        duration = _parse_int(row["duration_min"], f"{path.name}: duration_min")
        break_min = _parse_int(row["break_min"], f"{path.name}: break_min")
        crosses = _parse_int(row["crosses_midnight"], f"{path.name}: crosses_midnight")
        if crosses not in {0, 1}:
            raise ValidationError(
                f"{path.name}: crosses_midnight deve essere 0 o 1 (trovato {crosses})"
            )

        if sid in {"R", "SN", "F"}:
            if (
                duration != 0
                or break_min != 0
                or crosses != 0
                or row["start"]
                or row["end"]
            ):
                raise ValidationError(
                    f"{path.name}: turno {sid} deve avere duration_min=0, crosses_midnight=0 e start/end vuoti"
                )
            continue

        if duration <= 0:
            raise ValidationError(
                f"{path.name}: turno {sid} deve avere duration_min > 0"
            )

        if break_min < 0:
            raise ValidationError(
                f"{path.name}: break_min non può essere negativo per il turno {sid}"
            )

        for col in ("start", "end"):
            value = row[col]
            if len(value) != 5 or value[2] != ":":
                raise ValidationError(
                    f"{path.name}: valore '{value}' non valido in colonna {col} per turno {sid}"
                )
            hh, mm = value.split(":")
            if not (hh.isdigit() and mm.isdigit()):
                raise ValidationError(
                    f"{path.name}: orario non numerico '{value}' per turno {sid}"
                )
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValidationError(
                    f"{path.name}: orario fuori range '{value}' per turno {sid}"
                )

        start_min = int(row["start"].split(":")[0]) * 60 + int(row["start"].split(":")[1])
        end_min = int(row["end"].split(":")[0]) * 60 + int(row["end"].split(":")[1])
        raw_duration = (
            end_min - start_min if crosses == 0 else (24 * 60 - start_min + end_min)
        )
        if break_min >= raw_duration:
            raise ValidationError(
                f"{path.name}: break_min {break_min} per turno {sid} deve essere inferiore alla durata {raw_duration}"
            )
        expected_duration = raw_duration - break_min
        if duration != expected_duration:
            raise ValidationError(
                f"{path.name}: duration_min per turno {sid} deve essere {expected_duration} (ottenuto {duration})"
            )

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_shift_role(
    path: Path, employees: list[dict[str, str]], shifts: list[dict[str, str]]
) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    column_set = set(columns)

    if "shift_id" in column_set:
        shift_col = "shift_id"
    elif "shift_code" in column_set:
        shift_col = "shift_code"
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti per shift_role_eligibility (attese 'shift_id' o 'shift_code')"
        )

    if "ruolo" in column_set:
        role_col = "ruolo"
    elif "role" in column_set:
        role_col = "role"
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti per shift_role_eligibility (attese 'ruolo' o 'role')"
        )

    employee_roles = {row["ruolo"].strip().upper() for row in employees}
    shift_ids = {row["shift_id"].strip().upper() for row in shifts}

    seen_pairs = set()
    for idx, row in enumerate(rows, start=2):
        sid_raw = row.get(shift_col, "")
        role_raw = row.get(role_col, "")
        sid = sid_raw.strip().upper()
        role = role_raw.strip().upper()
        if not sid:
            raise ValidationError(f"{path.name}: {shift_col} vuoto alla riga {idx}")
        if not role:
            raise ValidationError(f"{path.name}: {role_col} vuoto alla riga {idx}")
        if sid not in shift_ids:
            raise ValidationError(
                f"{path.name}: turno '{sid}' non presente in shifts.csv"
            )
        if role not in employee_roles:
            raise ValidationError(
                f"{path.name}: ruolo '{role}' non presente in employees.csv"
            )

        row["shift_id"] = sid
        row["ruolo"] = role

        key = (sid, role)
        if key in seen_pairs:
            raise ValidationError(
                f"{path.name}: duplicato per coppia (shift_id, ruolo) {key}"
            )
        seen_pairs.add(key)

    demand_shifts = {"M", "P", "N"}
    for sid in sorted(demand_shifts & shift_ids):
        if not any(r["shift_id"] == sid for r in rows):
            raise ValidationError(
                f"{path.name}: nessun ruolo definito per il turno obbligatorio '{sid}'"
            )

    return rows, DatasetSummary(label=path.name, rows=len(rows))



def _check_month_plan(path: Path, shifts: list[dict[str, str]]) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    column_set = set(columns)

    if {"turno", "codice"}.issubset(column_set):
        shift_col = "turno"
        coverage_col = "codice"
        dept_col = "reparto" if "reparto" in column_set else None
    elif {"shift_code", "coverage_code"}.issubset(column_set):
        shift_col = "shift_code"
        coverage_col = "coverage_code"
        dept_col = "reparto_id" if "reparto_id" in column_set else None
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti per month_plan (attese 'turno/codice' oppure 'shift_code/coverage_code')"
        )

    required = {"data", shift_col, coverage_col}
    if dept_col:
        required.add(dept_col)
    _require_columns(columns, required, path.name)

    shift_ids = {row["shift_id"] for row in shifts}
    valid_turni = {"M", "P", "N"}

    for idx, row in enumerate(rows, start=2):
        shift_value = row[shift_col].strip()
        if not shift_value:
            raise ValidationError(
                f"{path.name}: {shift_col} vuoto alla riga {idx}"
            )
        if shift_value not in shift_ids:
            raise ValidationError(
                f"{path.name}: {shift_col} '{shift_value}' non presente in shifts.csv"
            )
        if shift_col == "turno" and shift_value not in valid_turni:
            raise ValidationError(
                f"{path.name}: turno '{shift_value}' non ammesso alla riga {idx}. Ammessi: {sorted(valid_turni)}"
            )

        coverage_value = row[coverage_col].strip()
        if not coverage_value:
            raise ValidationError(
                f"{path.name}: {coverage_col} vuoto alla riga {idx}"
            )

        if dept_col:
            dept_value = row[dept_col].strip()
            if not dept_value:
                raise ValidationError(
                    f"{path.name}: {dept_col} vuoto alla riga {idx}"
                )

        _parse_date(row["data"], f"{path.name}: data")

    if shift_col != "turno":
        for sid in sorted({"M", "P", "N"} & shift_ids):
            if not any(r[shift_col].strip() == sid for r in rows):
                raise ValidationError(
                    f"{path.name}: nessuna riga con shift_code '{sid}' per copertura obbligatoria"
                )

    return rows, DatasetSummary(label=path.name, rows=len(rows))




def _check_coverage_groups(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    column_set = set(columns)

    if {"codice", "turno", "gruppo"}.issubset(column_set):
        code_col = "codice"
        shift_col = "turno"
    elif {"coverage_code", "shift_code", "gruppo"}.issubset(column_set):
        code_col = "coverage_code"
        shift_col = "shift_code"
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti per coverage_groups (attese 'codice/turno' oppure 'coverage_code/shift_code')"
        )

    if "reparto_id" not in column_set:
        raise ValidationError(
            f"{path.name}: colonna mancante 'reparto_id' per coverage_groups"
        )

    dept_col = "reparto_id"

    required = {code_col, shift_col, dept_col, "gruppo", "total_staff", "ruoli_totale"}
    _require_columns(columns, required, path.name)

    seen = set()
    for idx, row in enumerate(rows, start=2):
        code = row[code_col].strip().upper()
        shift = row[shift_col].strip().upper()
        gruppo = row["gruppo"].strip().upper()
        dept = row[dept_col].strip().upper()

        if code_col != "codice":
            row["codice"] = code
        else:
            row["codice"] = code
        if shift_col != "turno":
            row["turno"] = shift
        else:
            row["turno"] = shift

        row["gruppo"] = gruppo
        row["reparto_id"] = dept
        row["reparto"] = dept

        if not code or not shift:
            raise ValidationError(
                f"{path.name}: codice o turno vuoto alla riga {idx}"
            )
        if not dept:
            raise ValidationError(
                f"{path.name}: reparto_id vuoto alla riga {idx}"
            )

        key = (row["codice"], row["turno"], row["reparto_id"], row["gruppo"])
        if key in seen:
            raise ValidationError(
                f"{path.name}: duplicato per (codice, turno, reparto_id, gruppo) {key}"
            )
        seen.add(key)
        total_staff = _parse_int(row["total_staff"], f"{path.name}: total_staff")
        if total_staff <= 0:
            raise ValidationError(
                f"{path.name}: total_staff deve essere positivo per {key}"
            )
        roles = [
            part.strip().upper() for part in row["ruoli_totale"].split("|") if part.strip()
        ]
        if not roles:
            raise ValidationError(
                f"{path.name}: ruoli_totale vuoto per {key}"
            )
        row["ruoli_totale_list"] = roles

    return rows, DatasetSummary(label=path.name, rows=len(rows))



def _check_coverage_roles(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    column_set = set(columns)

    if {"codice", "turno", "gruppo", "ruolo"}.issubset(column_set):
        code_col = "codice"
        shift_col = "turno"
    elif {"coverage_code", "shift_code", "gruppo", "ruolo"}.issubset(column_set):
        code_col = "coverage_code"
        shift_col = "shift_code"
    else:
        raise ValidationError(
            f"{path.name}: colonne mancanti per coverage_roles (attese 'codice/turno' oppure 'coverage_code/shift_code')"
        )

    if "reparto_id" not in column_set:
        raise ValidationError(
            f"{path.name}: colonna mancante 'reparto_id' per coverage_roles"
        )

    dept_col = "reparto_id"

    required = {code_col, shift_col, dept_col, "gruppo", "ruolo", "min_ruolo"}
    _require_columns(columns, required, path.name)

    seen = set()
    for idx, row in enumerate(rows, start=2):
        code = row[code_col].strip().upper()
        shift = row[shift_col].strip().upper()
        gruppo = row["gruppo"].strip().upper()
        role = row["ruolo"].strip().upper()
        dept = row[dept_col].strip().upper()

        row["codice"] = code
        row["turno"] = shift
        row["gruppo"] = gruppo
        row["ruolo"] = role
        row["reparto_id"] = dept
        row["reparto"] = dept

        if not dept:
            raise ValidationError(
                f"{path.name}: reparto_id vuoto alla riga {idx}"
            )

        key = (row["codice"], row["turno"], row["reparto_id"], row["gruppo"], role)
        if key in seen:
            raise ValidationError(
                f"{path.name}: duplicato per (codice, turno, reparto_id, gruppo, ruolo) {key}"
            )
        seen.add(key)
        _parse_int(row["min_ruolo"], f"{path.name}: min_ruolo")

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _validate_groups_vs_roles(
    groups: list[dict[str, str]],
    roles: list[dict[str, str]],
    eligibility_pairs: set[tuple[str, str]],
) -> None:
    groups_index = {
        (row["codice"], row["turno"], row["reparto_id"], row["gruppo"]): row
        for row in groups
    }
    for row in roles:
        key = (row["codice"], row["turno"], row["reparto_id"], row["gruppo"])
        if key not in groups_index:
            raise ValidationError(
                "coverage_roles.csv: combinazione "
                f"(codice={key[0]}, turno={key[1]}, reparto_id={key[2]}, gruppo={key[3]}) "
                "non definita in coverage_groups.csv"
            )
        pair = (row["turno"], row["ruolo"])
        if pair not in eligibility_pairs:
            raise ValidationError(
                "coverage_roles.csv: ruolo {1} non idoneo per il turno {0}".format(*pair)
            )

    for key, grp in groups_index.items():
        allowed = set(grp["ruoli_totale_list"])
        role_rows = [
            row
            for row in roles
            if (row["codice"], row["turno"], row["reparto_id"], row["gruppo"]) == key
        ]
        for row in role_rows:
            if row["ruolo"] not in allowed:
                raise ValidationError(
                    "coverage_roles.csv: ruolo {0} non presente in ruoli_totale per "
                    "(codice={1}, turno={2}, reparto_id={3}, gruppo={4})".format(
                        row["ruolo"], *key
                    )
                )
        sum_min = sum(int(float(row["min_ruolo"])) for row in role_rows)
        if sum_min > int(float(grp["total_staff"])):
            raise ValidationError(
                "Incoerenza: somma min_ruolo supera total_staff per "
                "(codice={0}, turno={1}, reparto_id={2}, gruppo={3})".format(*key)
            )

def _check_history(
    path: Path,
    employees: list[dict[str, str]],
    shifts: list[dict[str, str]],
) -> tuple[list[dict[str, str]], DatasetSummary]:
    if not path.exists():
        return [], DatasetSummary(label=path.name, rows=0)

    rows, columns = _read_csv_dicts(path)
    _require_columns(columns, {"data", "employee_id", "turno"}, path.name)

    known_employees = {row["employee_id"] for row in employees}
    known_shifts = {row["shift_id"] for row in shifts}

    seen_keys = set()
    per_day = defaultdict(set)
    for idx, row in enumerate(rows, start=2):
        _parse_date(row["data"], f"{path.name}: data")
        emp = row["employee_id"]
        turno = row["turno"]
        if emp not in known_employees:
            raise ValidationError(
                f"{path.name}: employee_id '{emp}' non presente in employees.csv"
            )
        if turno not in known_shifts:
            raise ValidationError(
                f"{path.name}: turno '{turno}' non presente in shifts.csv"
            )
        key = (row["data"], emp, turno)
        if key in seen_keys:
            raise ValidationError(
                f"{path.name}: duplicato per chiave {key}"
            )
        seen_keys.add(key)
        day_key = (row["data"], emp)
        if turno in per_day[day_key]:
            raise ValidationError(
                f"{path.name}: più di un turno per {emp} in data {row['data']}"
            )
        per_day[day_key].add(turno)

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_leaves(path: Path, employees: list[dict[str, str]]) -> tuple[list[dict[str, str]], DatasetSummary]:
    if not path.exists():
        return [], DatasetSummary(label=path.name, rows=0)
    rows, columns = _read_csv_dicts(path)
    column_set = set(columns)
    if "start_date" in column_set:
        start_col = "start_date"
    elif "date_from" in column_set:
        start_col = "date_from"
    else:
        raise ValidationError(
            f"{path.name}: colonna mancante per la data di inizio (attese 'start_date' o 'date_from')"
        )
    if "end_date" in column_set:
        end_col = "end_date"
    elif "date_to" in column_set:
        end_col = "date_to"
    else:
        raise ValidationError(
            f"{path.name}: colonna mancante per la data di fine (attese 'end_date' o 'date_to')"
        )
    if "tipo" in column_set:
        type_col = "tipo"
    elif "type" in column_set:
        type_col = "type"
    else:
        raise ValidationError(
            f"{path.name}: colonna mancante per il tipo di assenza (attese 'tipo' o 'type')"
        )
    _require_columns(columns, {"employee_id", start_col, end_col, type_col}, path.name)

    known_employees = {row["employee_id"] for row in employees}
    for idx, row in enumerate(rows, start=2):
        emp = row["employee_id"]
        if emp not in known_employees:
            raise ValidationError(
                f"{path.name}: employee_id '{emp}' non presente in employees.csv"
            )
        start_dt = _parse_date(row[start_col], f"{path.name}: {start_col}")
        end_dt = _parse_date(row[end_col], f"{path.name}: {end_col}")
        if end_dt < start_dt:
            raise ValidationError(
                f"{path.name}: intervallo negativo per employee_id {emp} ({row[start_col]} > {row[end_col]})"
            )
        row["start_date"] = row[start_col]
        row["end_date"] = row[end_col]
        row["tipo"] = row[type_col]
    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_availability(
    path: Path,
    employees: list[dict[str, str]],
    shifts: list[dict[str, str]],
    horizon_start: date,
    horizon_end: date,
) -> tuple[list[dict[str, str]], DatasetSummary]:
    if not path.exists():
        return [], DatasetSummary(label=path.name, rows=0)

    rows, columns = _read_csv_dicts(path)
    _require_columns(columns, {"data", "employee_id"}, path.name)

    known_employees = {row["employee_id"] for row in employees}
    allowed_turns = {row["shift_id"] for row in shifts if int(row["duration_min"] or 0) > 0}

    for idx, row in enumerate(rows, start=2):
        emp = row["employee_id"]
        if emp not in known_employees:
            raise ValidationError(
                f"{path.name}: employee_id '{emp}' non presente in employees.csv"
            )
        day = _parse_date(row["data"], f"{path.name}: data")
        if day < horizon_start or day > horizon_end:
            raise ValidationError(
                f"{path.name}: data {row['data']} fuori dall'orizzonte configurato"
            )
        turno = row.get("turno", "").upper()
        if turno and turno not in ("ALL", "*") and turno not in allowed_turns:
            raise ValidationError(
                f"{path.name}: turno '{turno}' non ammesso (ammessi: {sorted(allowed_turns)} o ALL/*/vuoto)"
            )

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_holidays(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    if not path.exists():
        return [], DatasetSummary(label=path.name, rows=0)
    rows, columns = _read_csv_dicts(path)
    if not rows:
        return rows, DatasetSummary(label=path.name, rows=0)
    _require_columns(columns, {"data"}, path.name)
    for row in rows:
        _parse_date(row["data"], f"{path.name}: data")
    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _simple_yaml_load(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(0, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_value = line.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValidationError("config: indentazione non valida")

        if key_value.endswith(":"):
            key = key_value[:-1].strip()
            new_dict: dict = {}
            stack[-1][1][key] = new_dict
            stack.append((indent + 2, new_dict))
            continue

        if ":" not in key_value:
            raise ValidationError(f"config: linea non riconosciuta: {key_value}")

        key, value = key_value.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not stack:
            raise ValidationError("config: struttura non valida")

        if value == "":
            new_dict = {}
            stack[-1][1][key] = new_dict
            stack.append((indent + 2, new_dict))
            continue

        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value
        stack[-1][1][key] = parsed_value

    return root


def run_checks(config_path: Path, data_dir: Path) -> ValidationResult:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = _simple_yaml_load(f.read())

    try:
        horizon_cfg = cfg["horizon"]
        horizon_start = _parse_date(str(horizon_cfg["start_date"]), "config: horizon.start_date")
        horizon_end = _parse_date(str(horizon_cfg["end_date"]), "config: horizon.end_date")
    except KeyError as exc:
        raise ValidationError(
            "config: sezione horizon incompleta (richiesti start_date e end_date)"
        ) from exc

    if horizon_end < horizon_start:
        raise ValidationError(
            "config: horizon.end_date deve essere >= horizon.start_date"
        )

    defaults = cfg.get("defaults", {}) or {}

    weeks_in_horizon = _count_weeks_in_horizon(horizon_start, horizon_end)
    if weeks_in_horizon <= 0:
        raise ValidationError(
            "config: orizzonte senza settimane valide (start_date > end_date?)"
        )
    horizon_days = (horizon_end - horizon_start).days + 1
    if horizon_days <= 0:
        raise ValidationError(
            "config: orizzonte senza giorni validi (start_date > end_date?)"
        )

    employees, emp_summary = _check_employees(
        data_dir / "employees.csv",
        defaults,
        cfg.get("roles", {}) or {},
        weeks_in_horizon,
        horizon_days,
    )
    shifts, shift_summary = _check_shifts(data_dir / "shifts.csv")
    shift_role_rows, shift_role_summary = _check_shift_role(
        data_dir / "shift_role_eligibility.csv", employees, shifts
    )
    month_plan, month_summary = _check_month_plan(data_dir / "month_plan.csv", shifts)
    coverage_groups, groups_summary = _check_coverage_groups(data_dir / "coverage_groups.csv")
    coverage_roles, roles_summary = _check_coverage_roles(data_dir / "coverage_roles.csv")

    eligibility_pairs = {(row["shift_id"], row["ruolo"]) for row in shift_role_rows}
    _validate_groups_vs_roles(coverage_groups, coverage_roles, eligibility_pairs)

    history, history_summary = _check_history(data_dir / "history.csv", employees, shifts)
    leaves, leaves_summary = _check_leaves(data_dir / "leaves.csv", employees)
    availability, availability_summary = _check_availability(
        data_dir / "availability.csv", employees, shifts, horizon_start, horizon_end
    )
    holidays, holidays_summary = _check_holidays(data_dir / "holidays.csv")

    summaries = [
        emp_summary,
        shift_summary,
        shift_role_summary,
        month_summary,
        groups_summary,
        roles_summary,
        history_summary,
        leaves_summary,
        availability_summary,
        holidays_summary,
    ]

    return ValidationResult(
        summaries=summaries,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esegue controlli di coerenza sui CSV senza dipendenze esterne."
    )
    parser.add_argument("--config", required=True, type=Path, help="Percorso al file config.yaml")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory contenente i file CSV",
    )
    args = parser.parse_args()

    try:
        result = run_checks(args.config, args.data_dir)
    except (ValidationError, FileNotFoundError) as exc:
        print('[FAIL] Controlli falliti:')
        print(str(exc))
        raise SystemExit(1)

    horizon_days = (result.horizon_end - result.horizon_start).days + 1
    print('[OK] Controlli completati con successo')
    print(f"Orizzonte configurato: {result.horizon_start} -> {result.horizon_end} ({horizon_days} giorni)")
    for summary in result.summaries:
        print(f"- {summary.label}: {summary.rows} righe")


if __name__ == "__main__":
    main()
