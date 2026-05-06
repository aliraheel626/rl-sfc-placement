"""
Single source of truth for eval metrics: multi-episode aggregation, comparison table,
and eval curve plots shared by *compare.py* and *model.py* (TrainingEvalCallback).

Add a new metric here only — then wire it into *run_eval* return dicts in compare.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

# Keys collected per algorithm per eval episode (run_eval result dict).
TRACKED_EVAL_METRIC_KEYS: tuple[str, ...] = (
    "acceptance_ratio",
    "latency_violation_ratio",
    "avg_sfc_tenancy",
    "avg_vnf_tenancy",
    "avg_substrate_utilization",
    "avg_sec_margin",
)


@dataclass(frozen=True)
class EvalPlotSpec:
    """One PNG curve plot for eval dashboards."""

    key: str
    filename: str
    ylabel: str
    title: str  # Shown as f"{title} ({num_requests} req/episode)"
    ma_color: str = "orange"
    ylim: Optional[tuple[float, float]] = None
    as_percent: bool = False


EVAL_PLOT_SPECS: tuple[EvalPlotSpec, ...] = (
    EvalPlotSpec(
        key="acceptance_ratio",
        filename="sfc_ppo_acceptance_ratio.png",
        ylabel="Acceptance Ratio (per episode)",
        title="Episode Acceptance Ratio vs Episode Number",
        ma_color="orange",
        ylim=(0.0, 1.05),
    ),
    EvalPlotSpec(
        key="avg_substrate_utilization",
        filename="substrate_util_training.png",
        ylabel="Substrate Utilization",
        title="Substrate Utilization vs Episode Number",
        ma_color="indigo",
        ylim=(0.0, 1.05),
        as_percent=True,
    ),
    EvalPlotSpec(
        key="avg_sfc_tenancy",
        filename="sfc_per_node_training.png",
        ylabel="Avg SFCs per Occupied Node",
        title="Avg SFCs per Occupied Node vs Episode Number",
        ma_color="darkblue",
    ),
    EvalPlotSpec(
        key="avg_vnf_tenancy",
        filename="vnf_per_node_training.png",
        ylabel="Avg VNFs per Occupied Node",
        title="Avg VNFs per Occupied Node vs Episode Number",
        ma_color="darkgreen",
    ),
    EvalPlotSpec(
        key="avg_sec_margin",
        filename="security_score_training.png",
        ylabel="Avg Security Margin (node score − req min)",
        title="Avg Security Margin vs Episode Number",
        ma_color="teal",
    ),
)


@dataclass(frozen=True)
class ComparisonTableRow:
    key: str
    label: str
    higher_is_better: bool


# PPO vs BestFit summary table (main() in compare.py).
COMPARISON_TABLE_ROWS: tuple[ComparisonTableRow, ...] = (
    ComparisonTableRow("acceptance_ratio", "Acceptance Ratio", True),
    ComparisonTableRow("avg_sec_margin", "Avg Security Margin", True),
    ComparisonTableRow("avg_sfc_tenancy", "Avg SFC Tenancy (SFCs/occupied node)", False),
    ComparisonTableRow("avg_vnf_tenancy", "Avg VNF Tenancy (VNFs/occupied node)", False),
)


def get_eval_metric(res: Mapping[str, Any], key: str) -> float:
    v = res.get(key, 0.0)
    return float(v) if v is not None else 0.0


def empty_series_by_algo() -> dict[str, list[float]]:
    return {k: [] for k in TRACKED_EVAL_METRIC_KEYS}


def append_eval_result_to_by_algo(
    by_algo: MutableMapping[str, MutableMapping[str, list]],
    algo_name: str,
    res: Mapping[str, Any],
) -> None:
    """Append one run_eval *res* for *algo_name* (mutates *by_algo*)."""
    if algo_name not in by_algo:
        by_algo[algo_name] = empty_series_by_algo()
    bucket = by_algo[algo_name]
    for key in TRACKED_EVAL_METRIC_KEYS:
        bucket[key].append(get_eval_metric(res, key))


def plot_eval_curves(
    episodes: list[int] | range,
    by_algo: Mapping[str, Mapping[str, list]],
    output_dir: str | Path,
    num_requests: int,
    *,
    ppo_highlight: str = "MaskablePPO",
    linewidth: float = 0.5,
    dpi: int = 150,
    verbose_print: bool = False,
    alpha: float = 0.6,
) -> None:
    """
    Write every plot in EVAL_PLOT_SPECS to *output_dir*.

    *by_algo* maps algorithm name → metric key → list of values (one per episode).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep_list = list(episodes)
    algos = list(by_algo.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(algos), 10)))[: len(algos)]
    color_map = dict(zip(algos, colors))

    def moving_avg(y: list[float], window: int):
        if len(y) < window or window < 2:
            return None, None
        k = np.ones(window) / window
        ma = np.convolve(np.asarray(y, dtype=np.float64), k, mode="valid")
        x = ep_list[window - 1 :]
        return x, ma

    for spec in EVAL_PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(10, 6))
        any_vals = False
        for algo in algos:
            series = by_algo.get(algo, {})
            vals = list(series.get(spec.key, []) or [])
            if not vals:
                continue
            any_vals = True
            ax.plot(
                ep_list,
                vals,
                alpha=alpha,
                label=algo,
                linewidth=linewidth,
                color=color_map[algo],
            )
            if algo == ppo_highlight and len(vals) > 10:
                window = min(50, len(vals) // 5)
                if window > 1:
                    x_ma, y_ma = moving_avg(vals, window)
                    if x_ma is not None and y_ma is not None:
                        ax.plot(
                            x_ma,
                            y_ma,
                            color=spec.ma_color,
                            linewidth=2,
                            label=f"Moving Avg ({window} ep)",
                        )
        if not any_vals:
            plt.close(fig)
            continue
        ax.set_title(f"{spec.title} ({num_requests} req/episode)", fontsize=13)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(spec.ylabel, fontsize=12)
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        if spec.as_percent:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        save_path = out / spec.filename
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        if verbose_print:
            print(f"Plot saved to: {save_path}")
