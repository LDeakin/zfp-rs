#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Generate local SVG comparison plots from Criterion benchmark output."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

SCALARS = ["i32", "i64", "f32", "f64"]
MODES = ["fixed_rate", "fixed_precision", "fixed_accuracy", "reversible"]
VARIANTS = [
    "zfp-rs",
    "zfp-rs-ffi",
    "zfp-sys",
    "zfp-rs-rayon2",
    "zfp-sys-omp2",
    "zfp-rs-ffi-omp2",
    "zfp-rs-rayon3",
    "zfp-sys-omp3",
    "zfp-rs-ffi-omp3",
]

# Main x-axis groups: base implementation
BASES = ["zfp-rs", "zfp-rs-ffi", "zfp-sys"]

# Colour per base implementation
BASE_COLOR = {
    "zfp-rs": "#2f6f73",
    "zfp-rs-ffi": "#8a3d58",
    "zfp-sys": "#7a6b2f",
}

# Variant -> (base, thread_level)
VARIANT_MAP: dict[str, tuple[str, str]] = {
    "zfp-rs": ("zfp-rs", "serial"),
    "zfp-rs-rayon2": ("zfp-rs", "2T"),
    "zfp-rs-rayon3": ("zfp-rs", "3T"),
    "zfp-rs-ffi": ("zfp-rs-ffi", "serial"),
    "zfp-rs-ffi-omp2": ("zfp-rs-ffi", "2T"),
    "zfp-rs-ffi-omp3": ("zfp-rs-ffi", "3T"),
    "zfp-sys": ("zfp-sys", "serial"),
    "zfp-sys-omp2": ("zfp-sys", "2T"),
    "zfp-sys-omp3": ("zfp-sys", "3T"),
}


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


def _variant_for(base: str, thread: str) -> str | None:
    for v in VARIANTS:
        if VARIANT_MAP.get(v) == (base, thread):
            return v
    return None


def _format_case(case: str) -> str:
    """'d2_fixed_rate' -> 'Fixed Rate 2D'."""
    parts = case.split("_")
    dim = parts[0]
    dim_str = f"{dim[1:] if dim.startswith('d') else dim}D"
    mode_name = " ".join(parts[1:]).replace("_", " ").title()
    return f"{mode_name} {dim_str}"


def plot_operation(
    results: list[BenchResult],
    operation: str,
    scalar: str,
    output_dir: Path,
) -> None:
    selected = [
        r for r in results if r.operation == operation and r.scalar == scalar
    ]
    if not selected:
        return

    cases = sorted({r.case for r in selected}, key=case_key)
    lookup = {(r.case, r.variant): r.melem_per_s for r in selected}

    n_cases = len(cases)
    n_bases = len(BASES)

    # Layout: x-axis = cases, each case has n_bases stacked bars side-by-side
    bar_width = 0.30
    case_gap = bar_width + 0.04
    case_group_width = n_bases * case_gap + 0.4
    fig_width = max(10, case_group_width * n_cases + 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)

    # Stack layers: (label, hatch)
    stack_layers = [("serial", None), ("2T", "//"), ("3T", "xxx")]

    # Legend 1: base implementations (colours)
    leg1_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASE_COLOR[base])
        for base in BASES
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
        for _, hatch in stack_layers
    ]

    # Draw bars
    for c_idx, case in enumerate(cases):
        group_x = c_idx * case_group_width

        for b_idx, base in enumerate(BASES):
            bx = group_x + b_idx * case_gap

            # Gather throughputs per layer for this base+case
            layer_vals: list[float] = []
            for layer_label, _hatch in stack_layers:
                v = _variant_for(base, layer_label)
                layer_vals.append(lookup.get((case, v), 0.0) if v else 0.0)

            # Draw stacked segments
            bottom = 0.0
            for layer_idx in range(len(stack_layers)):
                layer_label, hatch = stack_layers[layer_idx]
                height = layer_vals[layer_idx] - bottom
                if height < 0:
                    height = 0
                ax.bar(
                    bx,
                    height,
                    bar_width,
                    bottom=bottom,
                    color=BASE_COLOR[base],
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.3,
                )
                bottom = layer_vals[layer_idx]

    # X-axis ticks at middle implementation of each case group
    middle_idx = n_bases // 2
    middle_offset = middle_idx * case_gap
    ax.set_xticks([c * case_group_width + middle_offset for c in range(n_cases)])
    ax.set_xticklabels([_format_case(c) for c in cases], fontsize=9, rotation=0, ha="center")

    ax.set_title(f"ZFP {scalar} {operation} throughput")
    ax.set_ylabel("throughput (Melem/s)")

    # Figure-level legend placed outside the axes
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
        labels=[label for label, _ in stack_layers],
        frameon=True,
        fontsize=9,
        loc="outside upper right",
        ncol=3,
        title="threads",
    )

    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.savefig(output_dir / f"api_compare_{scalar}_{operation}.svg", format="svg")
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
    for scalar in SCALARS:
        for operation in sorted({r.operation for r in results}):
            plot_operation(results, operation, scalar, args.output_dir)


if __name__ == "__main__":
    main()
