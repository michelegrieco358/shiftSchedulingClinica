"""Quick data consistency checks without pandas."""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
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

    with path.open("r", encoding="utf-8", newline="") as f:
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


def _check_employees(path: Path, defaults: dict) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(
        columns,
        {
            "employee_id",
            "nome",
            "ruolo",
            "reparto",
            "ore_dovute_mese_h",
            "saldo_prog_iniziale_h",
        },
        path.name,
    )

    ids = [row["employee_id"] for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValidationError(
            f"{path.name}: employee_id duplicati trovati: {sorted(duplicates)}"
        )

    allowed_roles = _resolve_allowed_roles(defaults, {row["ruolo"] for row in rows})
    allowed_departments = _resolve_allowed_departments(defaults)

    for row in rows:
        role = row["ruolo"].strip()
        if role not in allowed_roles:
            raise ValidationError(
                f"{path.name}: ruolo non ammesso '{role}'. Ruoli ammessi: {allowed_roles}"
            )
        dept = row["reparto"].strip()
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
        _parse_float(ore, f"{path.name}: ore_dovute_mese_h")
        _parse_float(saldo, f"{path.name}: saldo_prog_iniziale_h", allow_negative=True)

        for col, default_val in (
            ("max_week_hours_h", defaults.get("max_week_hours_h", 60)),
            ("max_month_extra_h", defaults.get("max_month_extra_h", 40)),
            ("max_nights_week", defaults.get("max_nights_week", 3)),
            ("max_nights_month", defaults.get("max_nights_month", 8)),
        ):
            val = row.get(col, "") or str(default_val)
            _parse_float(val, f"{path.name}: {col}")

        for ytd_col in ("saturday_count_ytd", "sunday_count_ytd", "holiday_count_ytd"):
            val = row.get(ytd_col, "0") or "0"
            _parse_int(val, f"{path.name}: {ytd_col}")

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_shifts(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(
        columns,
        {"shift_id", "start", "end", "duration_min", "crosses_midnight"},
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
        crosses = _parse_int(row["crosses_midnight"], f"{path.name}: crosses_midnight")
        if crosses not in {0, 1}:
            raise ValidationError(
                f"{path.name}: crosses_midnight deve essere 0 o 1 (trovato {crosses})"
            )

        if sid in {"R", "SN", "F"}:
            if duration != 0 or crosses != 0 or row["start"] or row["end"]:
                raise ValidationError(
                    f"{path.name}: turno {sid} deve avere duration_min=0, crosses_midnight=0 e start/end vuoti"
                )
            continue

        if duration <= 0:
            raise ValidationError(
                f"{path.name}: turno {sid} deve avere duration_min > 0"
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

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_shift_role(
    path: Path, employees: list[dict[str, str]], shifts: list[dict[str, str]]
) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(columns, {"shift_id", "ruolo"}, path.name)

    employee_roles = {row["ruolo"] for row in employees}
    shift_ids = {row["shift_id"] for row in shifts}

    seen_pairs = set()
    for idx, row in enumerate(rows, start=2):
        sid = row["shift_id"]
        role = row["ruolo"]
        if not sid:
            raise ValidationError(f"{path.name}: shift_id vuoto alla riga {idx}")
        if not role:
            raise ValidationError(f"{path.name}: ruolo vuoto alla riga {idx}")
        if sid not in shift_ids:
            raise ValidationError(
                f"{path.name}: shift_id '{sid}' non presente in shifts.csv"
            )
        if role not in employee_roles:
            raise ValidationError(
                f"{path.name}: ruolo '{role}' non presente in employees.csv"
            )
        key = (sid, role)
        if key in seen_pairs:
            raise ValidationError(
                f"{path.name}: duplicato per coppia (shift_id, ruolo) {key}"
            )
        seen_pairs.add(key)

    demand_shifts = {"M", "P", "N"}
    for sid in sorted(demand_shifts & shift_ids):
        if not any(row["shift_id"] == sid for row in rows):
            raise ValidationError(
                f"{path.name}: nessun ruolo definito per il turno obbligatorio '{sid}'"
            )

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_month_plan(path: Path, shifts: list[dict[str, str]]) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(columns, {"data", "turno", "codice"}, path.name)

    valid_turni = {"M", "P", "N"}
    shift_ids = {row["shift_id"] for row in shifts}

    for idx, row in enumerate(rows, start=2):
        turno = row["turno"]
        if turno not in valid_turni:
            raise ValidationError(
                f"{path.name}: turno '{turno}' non ammesso alla riga {idx}. Ammessi: {sorted(valid_turni)}"
            )
        if turno not in shift_ids:
            raise ValidationError(
                f"{path.name}: turno '{turno}' non presente in shifts.csv"
            )
        _parse_date(row["data"], f"{path.name}: data")

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_coverage_groups(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(
        columns,
        {"codice", "turno", "gruppo", "total_min", "ruoli_totale"},
        path.name,
    )

    seen = set()
    for idx, row in enumerate(rows, start=2):
        key = (row["codice"], row["turno"], row["gruppo"])
        if key in seen:
            raise ValidationError(
                f"{path.name}: duplicato per (codice, turno, gruppo) {key}"
            )
        seen.add(key)
        total_min = _parse_int(row["total_min"], f"{path.name}: total_min")
        if total_min <= 0:
            raise ValidationError(
                f"{path.name}: total_min deve essere positivo per {key}"
            )
        roles = [part.strip() for part in row["ruoli_totale"].split("|") if part.strip()]
        if not roles:
            raise ValidationError(
                f"{path.name}: ruoli_totale vuoto per {key}"
            )
        row["ruoli_totale_list"] = roles

    return rows, DatasetSummary(label=path.name, rows=len(rows))


def _check_coverage_roles(path: Path) -> tuple[list[dict[str, str]], DatasetSummary]:
    rows, columns = _read_csv_dicts(path)
    _require_columns(
        columns,
        {"codice", "turno", "gruppo", "ruolo", "min_ruolo"},
        path.name,
    )

    seen = set()
    for row in rows:
        key = (row["codice"], row["turno"], row["gruppo"], row["ruolo"])
        if key in seen:
            raise ValidationError(
                f"{path.name}: duplicato per (codice, turno, gruppo, ruolo) {key}"
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
        (row["codice"], row["turno"], row["gruppo"]): row for row in groups
    }
    for row in roles:
        key = (row["codice"], row["turno"], row["gruppo"])
        if key not in groups_index:
            raise ValidationError(
                "coverage_roles.csv: combinazione (codice, turno, gruppo) non definita in coverage_groups.csv"
            )
        pair = (row["turno"], row["ruolo"])
        if pair not in eligibility_pairs:
            raise ValidationError(
                "coverage_roles.csv: ruolo {1} non idoneo per il turno {0}".format(*pair)
            )

    for key, grp in groups_index.items():
        allowed = set(grp["ruoli_totale_list"])
        role_rows = [row for row in roles if (row["codice"], row["turno"], row["gruppo"]) == key]
        for row in role_rows:
            if row["ruolo"] not in allowed:
                raise ValidationError(
                    "coverage_roles.csv: ruolo {0} non presente in ruoli_totale per {1}".format(
                        row["ruolo"], key
                    )
                )
        sum_min = sum(int(float(row["min_ruolo"])) for row in role_rows)
        if sum_min > int(float(grp["total_min"])):
            raise ValidationError(
                "Incoerenza: somma min_ruolo supera total_min per {0}".format(key)
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
    _require_columns(columns, {"employee_id", "start_date", "end_date", "tipo"}, path.name)

    known_employees = {row["employee_id"] for row in employees}
    for idx, row in enumerate(rows, start=2):
        emp = row["employee_id"]
        if emp not in known_employees:
            raise ValidationError(
                f"{path.name}: employee_id '{emp}' non presente in employees.csv"
            )
        start_dt = _parse_date(row["start_date"], f"{path.name}: start_date")
        end_dt = _parse_date(row["end_date"], f"{path.name}: end_date")
        if end_dt < start_dt:
            raise ValidationError(
                f"{path.name}: intervallo negativo per employee_id {emp} ({row['start_date']} > {row['end_date']})"
            )
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

    employees, emp_summary = _check_employees(data_dir / "employees.csv", defaults)
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
        print("❌ Controlli falliti:")
        print(str(exc))
        raise SystemExit(1)

    horizon_days = (result.horizon_end - result.horizon_start).days + 1
    print("✅ Controlli completati con successo")
    print(f"Orizzonte configurato: {result.horizon_start} → {result.horizon_end} ({horizon_days} giorni)")
    for summary in result.summaries:
        print(f"- {summary.label}: {summary.rows} righe")


if __name__ == "__main__":
    main()
