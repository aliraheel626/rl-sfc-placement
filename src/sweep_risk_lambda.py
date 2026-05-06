"""
Sweep risk-lambda values and report acceptance-vs-risk tradeoffs.

Usage (from repo root):
    python src/sweep_risk_lambda.py --lambdas 0.0,0.1,0.2,0.35,0.5

This script will:
1) clone the base config for each lambda value,
2) train a model for that config,
3) run compare_all() evaluation,
4) write summary files:
   - <out-dir>/summary.csv
   - <out-dir>/summary.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import yaml

from src.requests import load_config
from src.train import train
from src.compare import compare_all


def _parse_lambdas(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("No lambda values parsed from --lambdas")
    return vals


def _lambda_tag(value: float) -> str:
    """Filesystem-safe tag for a lambda value."""
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_sweep(
    base_config_path: str,
    lambdas: list[float],
    out_dir: str,
    timesteps: int | None,
    requests: int | None,
    seed: int,
    plot_freq: int,
    gnn_type: str,
    gnn_hidden_dim: int,
    gnn_features_dim: int,
    gnn_layers: int,
) -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(base_config_path)
    total_timesteps = timesteps or int(
        base_cfg.get("training", {}).get("total_timesteps", 1_000_000)
    )
    eval_requests = requests or int(
        base_cfg.get("evaluation", {}).get("num_requests", 1000)
    )

    rows: list[dict] = []

    for lam in lambdas:
        run_name = f"lambda_{_lambda_tag(lam)}"
        run_dir = out_path / run_name
        model_dir = run_dir / "models"
        logs_dir = run_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("risk", {})
        cfg["risk"]["enabled"] = True
        cfg["risk"]["lambda"] = float(lam)

        cfg_path = run_dir / "config.yaml"
        _write_yaml(cfg_path, cfg)

        save_path = str(model_dir / "sfc_ppo")
        print(f"\n=== Sweep run: lambda={lam} ===")
        print(f"Config: {cfg_path}")
        print(f"Save path: {save_path}")

        train(
            config_path=str(cfg_path),
            total_timesteps=total_timesteps,
            save_path=save_path,
            log_path=str(logs_dir),
            plot_freq=plot_freq,
            seed=seed,
            load_path=None,
            gnn_type=gnn_type,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_features_dim=gnn_features_dim,
            num_gnn_layers=gnn_layers,
        )

        best_model = model_dir / "sfc_ppo_best.zip"
        final_model = model_dir / "sfc_ppo.zip"
        model_path = str(best_model if best_model.exists() else final_model)

        results = compare_all(
            config_path=str(cfg_path),
            model_path=model_path,
            num_requests=eval_requests,
            save_plot=str(run_dir / "comparison.png"),
            verbose=False,
        )

        ppo = results.get("MaskablePPO", {})
        bestfit = results.get("BestFitPlacement", {})
        row = {
            "lambda": lam,
            "ppo_acceptance_ratio": ppo.get("acceptance_ratio", 0.0),
            "ppo_avg_risk_integral": ppo.get("avg_risk_integral", ppo.get("avg_risk_score", 0.0)),
            "ppo_avg_latency": ppo.get("avg_latency", 0.0),
            "bestfit_acceptance_ratio": bestfit.get("acceptance_ratio", 0.0),
            "bestfit_avg_risk_integral": bestfit.get("avg_risk_integral", bestfit.get("avg_risk_score", 0.0)),
            "delta_acceptance_vs_bestfit": ppo.get("acceptance_ratio", 0.0)
            - bestfit.get("acceptance_ratio", 0.0),
            "delta_risk_vs_bestfit": ppo.get("avg_risk_integral", ppo.get("avg_risk_score", 0.0))
            - bestfit.get("avg_risk_integral", bestfit.get("avg_risk_score", 0.0)),
            "run_dir": str(run_dir),
            "model_path": model_path,
        }
        rows.append(row)
        print(
            "Result: "
            f"acc={row['ppo_acceptance_ratio']:.4f}, "
            f"risk={row['ppo_avg_risk_integral']:.4f}, "
            f"delta_acc_vs_bestfit={row['delta_acceptance_vs_bestfit']:.4f}"
        )

    csv_path = out_path / "summary.csv"
    json_path = out_path / "summary.json"

    fieldnames = [
        "lambda",
        "ppo_acceptance_ratio",
        "ppo_avg_risk_integral",
        "ppo_avg_latency",
        "bestfit_acceptance_ratio",
        "bestfit_avg_risk_integral",
        "delta_acceptance_vs_bestfit",
        "delta_risk_vs_bestfit",
        "run_dir",
        "model_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote sweep summary: {csv_path}")
    print(f"Wrote sweep summary: {json_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep risk.lambda and report acceptance-vs-risk tradeoff"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Base config path"
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        required=True,
        help="Comma-separated values, e.g. 0.0,0.1,0.2,0.35,0.5",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sweeps/risk_lambda",
        help="Output directory for all runs",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training timesteps per lambda",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Override evaluation requests per lambda",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=10000,
        help="Training callback plot frequency",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="sage",
        choices=["gcn", "gat", "sage"],
        help="GNN layer type",
    )
    parser.add_argument(
        "--gnn-hidden-dim", type=int, default=64, help="GNN hidden dimension"
    )
    parser.add_argument(
        "--gnn-features-dim", type=int, default=256, help="GNN output features dim"
    )
    parser.add_argument(
        "--gnn-layers", type=int, default=3, help="Number of GNN layers"
    )

    args = parser.parse_args()
    lambdas = _parse_lambdas(args.lambdas)
    run_sweep(
        base_config_path=args.config,
        lambdas=lambdas,
        out_dir=args.out_dir,
        timesteps=args.timesteps,
        requests=args.requests,
        seed=args.seed,
        plot_freq=args.plot_freq,
        gnn_type=args.gnn_type,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_features_dim=args.gnn_features_dim,
        gnn_layers=args.gnn_layers,
    )


if __name__ == "__main__":
    main()

