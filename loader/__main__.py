from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Loader clinica - Step A (v6)")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument(
        "--export-csv", action="store_true", help="Esporta i DF espansi come CSV di debug"
    )
    ap.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Cartella di destinazione per i CSV di debug (default: <data-dir>/_expanded)",
    )
    args = ap.parse_args()

    try:
        from . import load_all
    except ModuleNotFoundError as exc:  # pragma: no cover - dipendenza runtime
        if getattr(exc, "name", None) == "pandas":
            raise SystemExit(
                "Errore: il modulo 'pandas' non è installato. "
                "Eseguire `pip install -r requirements.txt` prima di lanciare il loader."
            ) from exc
        raise

    data = load_all(args.config, args.data_dir)

    print("OK: caricati e costruiti i dati.")
    print(f"- employees: {len(data.employees_df)}")
    print(f"- month_plan righe: {len(data.month_plan_df)}")
    print(f"- shift slots: {len(data.shift_slots_df)}")
    print(f"- slot requirements: {len(data.slot_requirements_df)}")
    print(f"- coverage groups (espansi): {len(data.groups_total_expanded)}")
    print(f"- coverage role min (espansi): {len(data.groups_role_min_expanded)}")
    print(f"- calendar giorni: {len(data.calendar_df)}")
    print(f"- availability righe: {len(data.availability_df)}")
    print(f"- leaves righe: {len(data.leaves_df)}")
    print(f"- leave days: {len(data.leaves_days_df)}")
    print(f"- history righe: {len(data.history_df)}")
    print(f"- eligibility coppie (turno,ruolo): {len(data.eligibility_df)}")
    print(f"- gap pairs: {len(data.gap_pairs_df)}")
    if not data.holidays_df.empty:
        print(f"- holidays caricati: {len(data.holidays_df)}")

    if args.export_csv:
        default_outdir = os.path.join(args.data_dir, "_expanded")
        if args.export_dir is None:
            outdir = default_outdir
        else:
            outdir = args.export_dir
            if not os.path.isabs(outdir):
                outdir = os.path.join(args.data_dir, outdir)
        os.makedirs(outdir, exist_ok=True)
        data.calendar_df.to_csv(os.path.join(outdir, "calendar.csv"), index=False)
        data.month_plan_df.to_csv(os.path.join(outdir, "month_plan_with_calendar.csv"), index=False)
        data.shift_slots_df.to_csv(os.path.join(outdir, "shift_slots.csv"), index=False)
        data.slot_requirements_df.to_csv(
            os.path.join(outdir, "slot_requirements.csv"), index=False
        )
        data.groups_total_expanded.to_csv(
            os.path.join(outdir, "groups_total_expanded.csv"), index=False
        )
        data.groups_role_min_expanded.to_csv(
            os.path.join(outdir, "groups_role_min_expanded.csv"), index=False
        )
        data.employees_df.to_csv(os.path.join(outdir, "employees_processed.csv"), index=False)
        data.shifts_df.to_csv(os.path.join(outdir, "shifts_processed.csv"), index=False)
        data.availability_df.to_csv(
            os.path.join(outdir, "availability_with_calendar.csv"), index=False
        )
        data.leaves_df.to_csv(os.path.join(outdir, "leaves_expanded.csv"), index=False)
        data.leaves_days_df.to_csv(os.path.join(outdir, "leaves_days.csv"), index=False)
        data.history_df.to_csv(os.path.join(outdir, "history_with_calendar.csv"), index=False)
        data.eligibility_df.to_csv(
            os.path.join(outdir, "shift_role_eligibility_processed.csv"), index=False
        )
        data.gap_pairs_df.to_csv(os.path.join(outdir, "gap_pairs.csv"), index=False)
        print(f"Esportati CSV di debug in: {outdir}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
