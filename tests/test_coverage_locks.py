from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from loader.coverage import build_slot_requirements, expand_requirements
from loader.locks import validate_locks


def test_expand_requirements_combines_groups_and_roles() -> None:
    month_plan = pd.DataFrame(
        {
            "data": ["2024-01-01"],
            "data_dt": pd.to_datetime(["2024-01-01"]),
            "reparto_id": ["card"],
            "shift_code": ["m"],
            "coverage_code": ["cov1"],
        }
    )
    groups = pd.DataFrame(
        {
            "coverage_code": ["COV1"],
            "shift_code": ["M"],
            "reparto_id": ["CARD"],
            "gruppo": ["G1"],
            "total_staff": [4],
            "overstaff_cap": [""],
            "overstaff_cap_effective": [2],
            "ruoli_totale_list": [["INFERMIERE", "OSS"]],
        }
    )
    roles = pd.DataFrame(
        {
            "coverage_code": ["COV1", "COV1"],
            "shift_code": ["M", "M"],
            "reparto_id": ["CARD", "CARD"],
            "gruppo": ["G1", "G1"],
            "role": ["INFERMIERE", "OSS"],
            "min_ruolo": [2, 1],
        }
    )

    groups_out, roles_out = expand_requirements(month_plan, groups, roles)

    assert len(groups_out) == 1
    assert groups_out.loc[0, "ruoli_totale_set"] == "INFERMIERE|OSS"
    assert groups_out.loc[0, "total_staff"] == 4
    assert list(roles_out["role"]) == ["INFERMIERE", "OSS"]
    assert list(roles_out["min_ruolo"]) == [2, 1]


def test_build_slot_requirements_missing_demand_raises() -> None:
    slots_df = pd.DataFrame(
        {
            "slot_id": [1],
            "coverage_code": ["COV1"],
            "shift_code": ["M"],
            "reparto_id": ["CARD"],
        }
    )
    coverage_roles_df = pd.DataFrame(
        {
            "coverage_code": ["COV2"],
            "shift_code": ["M"],
            "reparto_id": ["CARD"],
            "role": ["INFERMIERE"],
            "min_ruolo": [1],
        }
    )

    with pytest.raises(ValueError, match="slots privi di requisiti"):
        build_slot_requirements(slots_df, coverage_roles_df)


def test_validate_locks_respects_cross_reparto_flag() -> None:
    pre_df = pd.DataFrame({"employee_id": ["E1"], "slot_id": [1], "lock": [1]})
    employees = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "reparto_id": ["CARD"],
            "role": ["INFERMIERE"],
        }
    )
    shift_slots = pd.DataFrame(
        {
            "slot_id": [1],
            "reparto_id": ["ER"],
            "shift_code": ["M"],
            "start_dt": [pd.Timestamp("2024-01-01 08:00:00")],
        }
    )

    with pytest.raises(ValueError, match="cross-reparto disabilitato"):
        validate_locks(
            pre_df,
            employees,
            shift_slots,
            cross_reparto_enabled=False,
        )

    with pytest.warns(UserWarning, match="cross-reparto abilitato"):
        result = validate_locks(
            pre_df,
            employees,
            shift_slots,
            cross_reparto_enabled=True,
        )

    assert result.loc[0, "reparto_id"] == "ER"
    assert result.loc[0, "shift_code"] == "M"
    assert result.loc[0, "date"] == date(2024, 1, 1)