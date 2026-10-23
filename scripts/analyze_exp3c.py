# scores experiment 3c (compaction, position and attribution in 3b's effect) and the GLM replication against their bars.
# uv run -m scripts.analyze_exp3c --run_root results/raw --output_dir results/raw/261023_exp3c_analysis_v1 --model_id openrouter/moonshotai/kimi-k3 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.generation.continuations import BENIGN_PATH
from src.metrics.adoption import adoption_rows
from src.metrics.io import write_json
from src.metrics.stats import cluster_bootstrap_diff, cluster_bootstrap_rate, holm
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
BUDGET = 30
BAR = 0.20
K3 = {
    "summary_self": "261022_exp3b_k3_summary_benign_v1",
    "message_user": "261022_exp3b_k3_user_benign_v1",
    "think_self": "261022_exp3b_k3_self_benign_v1",
    "full_message_user": "261023_exp3c_k3_full_user_benign_v1",
    "full_think_self": "261023_exp3c_k3_full_self_benign_v1",
    "summary_user": "261023_exp3c_k3_summary_userattr_benign_v1",
    "message_self": "261023_exp3c_k3_message_selfattr_benign_v1",
}
GLM = {
    "baseline": "261023_exp3b_glm53_summary_v1",
    "summary_self": "261023_exp3b_glm53_summary_benign_v1",
    "message_user": "261023_exp3b_glm53_user_benign_v1",
    "think_self": "261023_exp3b_glm53_self_benign_v1",
}


@dataclass
class Exp3cConfig(Config):
    run_root: str = "results/raw"


def averaged_diff(pairs: list[tuple[list[dict], list[dict]]], seed: int, n_resamples: int = 10000) -> dict:
    """Mean of several a-minus-b differences, resampling transcripts with one shared draw for every cell."""
    clusters = sorted({r["instance_id"] for a, b in pairs for r in a + b})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_resamples, len(clusters)))

    def cell(rows):
        sums = np.array([sum(r["adopted"] for r in rows if r["instance_id"] == c) for c in clusters], float)
        counts = np.array([sum(1 for r in rows if r["instance_id"] == c) for c in clusters], float)
        return sums, counts

    point, boot = [], []
    for a, b in pairs:
        (sa, ca), (sb, cb) = cell(a), cell(b)
        point.append(sa.sum() / ca.sum() - sb.sum() / cb.sum())
        boot.append(sa[draws].sum(1) / ca[draws].sum(1) - sb[draws].sum(1) / cb[draws].sum(1))
    diffs = np.mean(boot, axis=0)
    return {
        "diff": float(np.mean(point)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
    }


def load(root: Path, runs: dict) -> dict:
    rows = {
        cell: adoption_rows(root / run, cell, BENIGN_PATH, BUDGET)
        for cell, run in runs.items()
        if (root / run).exists()
    }
    return {cell: [r for r in rs if not r["is_failed"]] for cell, rs in rows.items()}


def rates(cells: dict, seed: int) -> dict:
    return {
        c: {
            "k": sum(r["adopted"] for r in rs),
            "n": len(rs),
            **cluster_bootstrap_rate(rs, "instance_id", "adopted", seed=seed),
        }
        for c, rs in cells.items()
    }


def main():
    config = simple_parsing.parse(Exp3cConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    root = Path(config.run_root)

    k3 = load(root, K3)
    comparisons = {
        "compaction_own_note": cluster_bootstrap_diff(
            k3["full_think_self"], k3["think_self"], "instance_id", "adopted", seed=config.seed
        ),
        "compaction_user": cluster_bootstrap_diff(
            k3["full_message_user"], k3["message_user"], "instance_id", "adopted", seed=config.seed
        ),
        "position": averaged_diff(
            [(k3["message_self"], k3["summary_self"]), (k3["message_user"], k3["summary_user"])], config.seed
        ),
        "attribution": averaged_diff(
            [(k3["summary_user"], k3["summary_self"]), (k3["message_user"], k3["message_self"])], config.seed
        ),
        "interaction_half_contrast": averaged_diff(
            [(k3["message_user"], k3["summary_user"]), (k3["summary_self"], k3["message_self"])], config.seed
        ),
    }
    names = list(comparisons)
    for name, p in zip(names, holm([comparisons[n]["p_two_sided"] for n in names])):
        comparisons[name]["p_holm"] = p
        if name != "interaction_half_contrast":
            comparisons[name]["cleared"] = abs(comparisons[name]["diff"]) >= BAR and (
                comparisons[name]["ci_low"] > 0 or comparisons[name]["ci_high"] < 0
            )

    glm = load(root, GLM)
    glm_result = None
    if len(glm) == len(GLM):
        d = cluster_bootstrap_diff(glm["message_user"], glm["summary_self"], "instance_id", "adopted", seed=config.seed)
        g = rates(glm, config.seed)
        glm_result = {
            "rates": g,
            "user_minus_summary": d,
            "positive_control_ok": g["message_user"]["rate"] >= 0.80,
            "baseline_ok": g["baseline"]["rate"] <= 0.05,
            "replicates": d["diff"] >= BAR and d["ci_low"] > 0,
        }
    result = {"k3_cells": rates(k3, config.seed), "k3_comparisons": comparisons, "glm": glm_result, "bar": BAR}
    path = write_json(result, output_dir / "analysis.json")
    for c, r in result["k3_cells"].items():
        print(f"K3 {c}: {r['k']}/{r['n']} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
    for n, c in comparisons.items():
        print(
            f"{n}: {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p_holm {c['p_holm']:.4f} cleared={c.get('cleared')}"
        )
    if glm_result:
        for c, r in glm_result["rates"].items():
            print(f"GLM {c}: {r['k']}/{r['n']}")
        print(
            f"GLM user - summary: {glm_result['user_minus_summary']['diff']:+.3f}; replicates={glm_result['replicates']}"
        )
    print(f"Results saved to {path}")


main()
