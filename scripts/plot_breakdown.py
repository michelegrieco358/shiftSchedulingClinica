from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


LINE_PATTERN = re.compile(
    r"^- (?P<component>[^:]+):\s+contributo=.*?\((?P<contrib_pct>[\d\.]+)%\)"
)
VIOL_EQ_PATTERN = re.compile(r"violazioni_equivalenti=[^()]+?\((?P<pct>[\d\.]+)%\)")
VIOL_PATTERN = re.compile(r"violazioni=[^()]+?\((?P<pct>[\d\.]+)%\)")


def _parse_breakdown(text: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    contrib_data: list[tuple[str, float]] = []
    violation_data: list[tuple[str, float]] = []

    for line in text.splitlines():
        if not line.startswith("- "):
            continue

        contrib_match = LINE_PATTERN.search(line)
        if not contrib_match:
            continue

        component = contrib_match.group("component")
        contrib_pct = float(contrib_match.group("contrib_pct"))
        contrib_data.append((component, contrib_pct))

        viol_match = VIOL_EQ_PATTERN.search(line) or VIOL_PATTERN.search(line)
        if viol_match:
            violation_data.append((component, float(viol_match.group("pct"))))

    return contrib_data, violation_data


def _plot_bar(
    data: Sequence[tuple[str, float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not data:
        raise ValueError("Nessun dato da rappresentare.")

    components, values = zip(*data)
    fig_width = max(6.0, 0.75 * len(values))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    bars = ax.bar(components, values, color="#4472c4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(args: Iterable[str]) -> None:
    args = list(args)
    input_path = Path(args[0]) if args else Path("objective_breakdown.txt")
    text = input_path.read_text(encoding="utf-8")

    contrib_data, violation_data = _parse_breakdown(text)

    _plot_bar(
        contrib_data,
        "Contributo percentuale per componente",
        "Contributo (%)",
        Path("reports/breakdown_contributions.png"),
    )
    _plot_bar(
        violation_data,
        "Violazioni equivalenti per componente",
        "Violazioni (%)",
        Path("reports/breakdown_violations.png"),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
