#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Generate local SVG comparison plots from Criterion benchmark output.

There is one plot per operation, holding every scalar type.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

SCALARS = ["i32", "i64", "f32", "f64"]
MODES = ["fixed_rate", "fixed_precision", "fixed_accuracy", "reversible"]

# `zfp-rs` and `zfp-rs-ffi` differ only by noise, so they share a base, and
# their throughputs are averaged into one bar
RS = "zfp-rs/zfp-rs-ffi"

# Main groups: base implementation
BASES = [RS, "zfp-sys"]

# Colour per base implementation
BASE_COLOR = {
    RS: "#2f6f73",
    "zfp-sys": "#7a6b2f",
}

# Variant -> (base, thread_level)
VARIANT_MAP: dict[str, tuple[str, str]] = {
    "zfp-rs": (RS, "serial"),
    "zfp-rs-rayon2": (RS, "2T"),
    "zfp-rs-ffi": (RS, "serial"),
    "zfp-rs-ffi-omp2": (RS, "2T"),
    "zfp-sys": ("zfp-sys", "serial"),
    "zfp-sys-omp2": ("zfp-sys", "2T"),
}

# Layout: horizontal bars, one block per case.
# Each block has a header row, then a group of one stacked bar per base
# implementation for each scalar type.
# Distances along the y-axis are in bar pitches.
BAR_HEIGHT = 0.88
GROUP_GAP = 0.8
BLOCK_HEADER = 2.0

# Stack layers: (label, hatch)
STACK_LAYERS = [("serial", None), ("2T", "//")]

# Figure size in inches
FIG_WIDTH = 10
INCHES_PER_PITCH = 0.1
# Room for the title, legends and both x-axes
MARGIN_HEIGHT = 1.8


@dataclass(frozen=True)
class BenchResult:
    operation: str
    scalar: str
    case: str
    variant: str
    mean_ns: float
    elements: int

    @property
    def melem_per_s(self) -> float:
        return self.elements * 1_000.0 / self.mean_ns


def throughput_elements(benchmark: dict) -> int:
    throughput = benchmark.get("throughput")
    if isinstance(throughput, dict):
        value = throughput.get("Elements")
        if isinstance(value, int):
            return value
    return 0


def split_case(case: str) -> tuple[str, str]:
    scalar, rest = case.split("_d", 1)
    return scalar, f"d{rest}"


def case_key(case: str) -> tuple[int, int]:
    dim, mode = case.split("_", 1)
    dim_number = int(dim.removeprefix("d"))
    return dim_number, MODES.index(mode)


def load_results(criterion_dir: Path) -> list[BenchResult]:
    results: list[BenchResult] = []
    for benchmark_json in criterion_dir.rglob("new/benchmark.json"):
        estimates_json = benchmark_json.with_name("estimates.json")
        if not estimates_json.exists():
            continue

        benchmark = json.loads(benchmark_json.read_text())
        estimates = json.loads(estimates_json.read_text())
        function_id = benchmark.get("function_id")
        value = benchmark.get("value_str")
        if not function_id or not value:
            continue

        parts = value.split("/")
        if len(parts) != 2:
            continue
        scalar, case = split_case(parts[0])

        mean = estimates.get("mean", {}).get("point_estimate")
        if mean is None:
            continue

        results.append(
            BenchResult(
                operation=function_id,
                scalar=scalar,
                case=case,
                variant=parts[1],
                mean_ns=float(mean),
                elements=throughput_elements(benchmark),
            )
        )
    return sorted(
        results,
        key=lambda item: (item.operation, item.scalar, item.case, item.variant),
    )


def write_summary(results: list[BenchResult], output_dir: Path) -> None:
    rows = ["operation,scalar,case,variant,mean_ns,elements,melem_per_s"]
    for result in results:
        rows.append(
            f"{result.operation},{result.scalar},{result.case},{result.variant},"
            f"{result.mean_ns:.6f},{result.elements},{result.melem_per_s:.6f}"
        )
    (output_dir / "api_compare.csv").write_text("\n".join(rows) + "\n")


def _variants_for(base: str, thread: str) -> list[str]:
    return [v for v, key in VARIANT_MAP.items() if key == (base, thread)]


def _format_case(case: str) -> str:
    """'d2_fixed_rate' -> 'Fixed Rate 2D'."""
    parts = case.split("_")
    dim = parts[0]
    dim_str = f"{dim[1:] if dim.startswith('d') else dim}D"
    mode_name = " ".join(parts[1:]).replace("_", " ").title()
    return f"{mode_name} {dim_str}"


def draw_group(ax, throughput: dict[str, float], top: float) -> None:
    """Draw one stacked bar per base implementation, the first starting at `top`.

    `throughput` maps a variant to its throughput for one scalar type and case.
    """
    for b_idx, base in enumerate(BASES):
        by = top + b_idx + 0.5

        # Gather throughputs per layer for this base, averaging its variants
        layer_vals: list[float] = []
        for layer_label, _hatch in STACK_LAYERS:
            values = [
                throughput[v]
                for v in _variants_for(base, layer_label)
                if v in throughput
            ]
            layer_vals.append(sum(values) / len(values) if values else 0.0)

        # Draw stacked segments
        start = 0.0
        for layer_idx in range(len(STACK_LAYERS)):
            layer_label, hatch = STACK_LAYERS[layer_idx]
            width = layer_vals[layer_idx] - start
            # Skip empty segments (a missing thread level, e.g. parallel
            # decompression outside fixed rate).
            # Matplotlib keeps a bar's autoscale margin from crossing the
            # bar's start, so an empty segment starting at the end of the
            # longest bar removes the margin beyond it and clips the bar.
            if width > 0:
                ax.barh(
                    by,
                    width,
                    BAR_HEIGHT,
                    left=start,
                    color=BASE_COLOR[base],
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.3,
                )
            start = layer_vals[layer_idx]


def plot_operation(
    results: list[BenchResult],
    operation: str,
    output_dir: Path,
) -> None:
    """Plot every scalar type of one operation on a single set of axes."""
    selected = [r for r in results if r.operation == operation]
    scalars = [s for s in SCALARS if any(r.scalar == s for r in selected)]
    if not scalars:
        return

    cases = sorted({r.case for r in selected}, key=case_key)
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for r in selected:
        lookup.setdefault((r.scalar, r.case), {})[r.variant] = r.melem_per_s
    group_pitch = len(BASES) + GROUP_GAP

    # A type with no results for a case, such as an integer type in fixed
    # accuracy, which is not benchmarked, gets no group in that case's block
    block_scalars = [[s for s in scalars if (s, case) in lookup] for case in cases]
    block_pitches = [BLOCK_HEADER + len(ss) * group_pitch for ss in block_scalars]
    total = sum(block_pitches) - GROUP_GAP

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, total * INCHES_PER_PITCH + MARGIN_HEIGHT),
        constrained_layout=True,
    )

    yticks: list[float] = []
    ylabels: list[str] = []
    block_top = 0.0
    for c_idx, case in enumerate(cases):
        # Block header, below a rule that separates it from the previous block
        if c_idx > 0:
            ax.axhline(block_top - GROUP_GAP / 2, color="#d0d0d0", linewidth=0.7)
        ax.text(
            0.005,
            block_top + BLOCK_HEADER / 2,
            _format_case(case),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        for s_idx, scalar in enumerate(block_scalars[c_idx]):
            top = block_top + BLOCK_HEADER + s_idx * group_pitch
            draw_group(ax, lookup[(scalar, case)], top)
            yticks.append(top + len(BASES) / 2)
            ylabels.append(scalar)

        block_top += block_pitches[c_idx]

    # Y-axis ticks at the middle implementation of each group
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_ylim(total, 0)
    ax.set_xlim(left=0)

    # The plot is tall, so label the throughput axis at both ends
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("throughput (Melem/s)", fontsize=10)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.suptitle(f"ZFP {operation} throughput")

    # Legend 1: base implementations (colours)
    leg1_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASE_COLOR[base]) for base in BASES
    ]

    # Legend 2: thread levels (hatch patterns)
    leg2_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#aaa",
            hatch=hatch,
            edgecolor="#555",
            linewidth=0.6,
        )
        for _, hatch in STACK_LAYERS
    ]

    # Figure-level legends placed outside the axes
    fig.legend(
        handles=leg1_handles,
        labels=BASES,
        frameon=True,
        fontsize=9,
        loc="outside upper left",
        ncol=3,
        title="Implementation",
    )
    fig.legend(
        handles=leg2_handles,
        labels=[label for label, _ in STACK_LAYERS],
        frameon=True,
        fontsize=9,
        loc="outside upper right",
        ncol=2,
        title="threads",
    )

    fig.savefig(output_dir / f"api_compare_{operation}.svg", format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Criterion output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory for generated local SVG and CSV files.",
    )
    args = parser.parse_args()

    results = load_results(args.criterion_dir)
    if not results:
        raise SystemExit(
            f"no Criterion benchmark results found in {args.criterion_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(results, args.output_dir)
    for operation in sorted({r.operation for r in results}):
        plot_operation(results, operation, args.output_dir)


if __name__ == "__main__":
    main()
