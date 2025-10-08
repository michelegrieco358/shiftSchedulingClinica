"""Utility functions to manage department pools and allowed departments."""

from __future__ import annotations

import pandas as pd


def load_pools(path: str) -> pd.DataFrame:
    """Load and validate department pools from ``pools.csv``.

    Parameters
    ----------
    path:
        Path to the ``pools.csv`` file.

    Returns
    -------
    pandas.DataFrame
        Normalised dataframe with ``["pool_id", "reparto_id"]`` columns.

    Raises
    ------
    ValueError
        If the input data contains null/empty values or if a ``reparto_id``
        belongs to more than one pool.
    """

    df = pd.read_csv(path, dtype=str)

    expected_cols = ["pool_id", "reparto_id"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "pools.csv: colonne mancanti: " + ", ".join(sorted(missing_cols))
        )

    df = df.loc[:, expected_cols].copy()

    if df.isna().any().any():
        null_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "pools.csv: valori null non ammessi nelle colonne: "
            + ", ".join(sorted(null_cols))
        )

    for col in expected_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()

    if (df == "").any().any():
        empty_cols = df.columns[(df == "").any()].tolist()
        raise ValueError(
            "pools.csv: valori vuoti non ammessi nelle colonne: "
            + ", ".join(sorted(empty_cols))
        )

    df = df.drop_duplicates().reset_index(drop=True)

    duplicated_departments = (
        df.groupby("reparto_id")["pool_id"].nunique().loc[lambda s: s > 1].index
    )
    if not duplicated_departments.empty:
        duplicates = sorted(duplicated_departments.tolist())
        raise ValueError(
            "pools.csv: i seguenti reparti appartengono a più pool: "
            + ", ".join(duplicates)
        )

    return df.sort_values(["pool_id", "reparto_id"]).reset_index(drop=True)


def derive_allowed_reparti(
    employees: pd.DataFrame, pools: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the list of allowed departments for every employee.

    Parameters
    ----------
    employees:
        DataFrame containing at least ``employee_id`` and ``reparto_id``.
        The input is not mutated.
    pools:
        DataFrame with ``pool_id`` and ``reparto_id`` columns, typically
        returned by :func:`load_pools`.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``employees_enriched`` and ``employee_allowed_reparti`` dataframes. La
        lista ``allowed_reparti`` contiene il reparto di origine per primo,
        seguito dagli altri reparti del pool in ordine alfabetico.
    """

    for column in ["employee_id", "reparto_id"]:
        if column not in employees.columns:
            raise ValueError(
                "employees: colonna mancante: " + column
            )

    for column in ["pool_id", "reparto_id"]:
        if column not in pools.columns:
            raise ValueError("pools: colonna mancante: " + column)

    pools_normalized = pools.copy()
    for col in ["pool_id", "reparto_id"]:
        pools_normalized[col] = (
            pools_normalized[col].astype(str).str.strip().str.upper()
        )

    dept_to_pool = pools_normalized.set_index("reparto_id")["pool_id"].to_dict()

    pool_to_departments: dict[str, list[str]] = {}
    for pool_id, pool_df in pools_normalized.groupby("pool_id"):
        pool_departments = pool_df["reparto_id"].tolist()
        seen: set[str] = set()
        ordered: list[str] = []
        for dept in pool_departments:
            if dept not in seen:
                ordered.append(dept)
                seen.add(dept)
        pool_to_departments[pool_id] = ordered

    home_departments = employees["reparto_id"].astype(str).str.strip().str.upper()

    allowed_lists: list[list[str]] = []
    for home in home_departments:
        if home == "":
            allowed_lists.append([])
            continue

        allowed = [home]
        pool_id = dept_to_pool.get(home)
        if pool_id:
            pool_depts = {
                dept for dept in pool_to_departments.get(pool_id, []) if dept != home
            }
            allowed.extend(sorted(pool_depts))
        allowed_lists.append(allowed)

    employees_enriched = employees.copy()
    employees_enriched["allowed_reparti"] = allowed_lists

    exploded = employees_enriched[["employee_id", "allowed_reparti"]].explode(
        "allowed_reparti"
    )
    exploded = exploded.dropna(subset=["allowed_reparti"])
    exploded = exploded.loc[exploded["allowed_reparti"] != ""]

    employee_allowed_reparti = (
        exploded.rename(columns={"allowed_reparti": "reparto_id_allowed"})
        .reset_index(drop=True)
    )

    return employees_enriched, employee_allowed_reparti

