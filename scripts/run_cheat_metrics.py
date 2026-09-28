# rescores every arm's saved diffs with the current diff check and reports per-arm rates with cluster bootstrap intervals over tasks.
# uv run -m scripts.run_cheat_metrics --dataset_path data/processed/tasks.jsonl --run_dirs results/raw/260928_pilot_small_accurate_v1 results/raw/260928_pilot_small_planted_self_v1 --output_dir results/raw/260928_pilot_small_metrics_v1 --seed 42

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.diff_check import check_diff
from src.metrics.io import write_json
from src.utils.config import write_config_json
from src.utils.git import get_git_hash

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    dataset_path: str = ""
    run_dirs: list[str] = field(default_factory=list)
    output_dir: str = ""
    n_boot: int = 10000
    seed: int = 42


def rescore(run_dir: Path, tasks: dict[str, dict]) -> tuple[str, list[dict]]:
    config = json.loads((run_dir / "config.json").read_text())
    rows = []
    for row in load_jsonl(run_dir / "outputs.jsonl"):
        task = tasks[row["instance_id"]]
        check = (
            check_diff(row["diff"], task[f"{config['split']}_test_patch"], task["patch"])
            if row["diff"] is not None
            else None
        )
        rows.append(
            {
                "id": row["id"],
                "instance_id": row["instance_id"],
                "failed": row["diff"] is None,
                "is_cheat": check.is_cheat if check else None,
                "is_scratch": bool(check.scratch) if check else None,
                "needs_review": check.needs_review if check else None,
                "f2p_passed": row["f2p_passed"],
                "hit_limit": row["n_messages"] >= config["message_limit"],
                "flags": {k: v for k, v in check.to_dict().items() if v and k not in ("is_cheat", "needs_review")}
                if check
                else None,
            }
        )
    return config["arm"], rows


def per_task(rows: list[dict], key: str, task_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(task_ids))
    counts = np.zeros(len(task_ids))
    for row in rows:
        if row[key] is None:
            continue
        i = task_ids.index(row["instance_id"])
        sums[i] += row[key]
        counts[i] += 1
    return sums, counts


def bootstrap_rates(arms: dict[str, list[dict]], key: str, task_ids: list[str], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(task_ids), size=(n_boot, len(task_ids)))
    stats = {arm: per_task(rows, key, task_ids) for arm, rows in arms.items()}
    # the same task resample is applied to every arm, so differences are paired over tasks
    boot = {arm: sums[idx].sum(1) / np.maximum(counts[idx].sum(1), 1) for arm, (sums, counts) in stats.items()}
    out = {}
    for arm, (sums, counts) in stats.items():
        out[arm] = {
            "rate": sums.sum() / counts.sum(),
            "ci95": list(np.percentile(boot[arm], [2.5, 97.5])),
            "n": int(counts.sum()),
        }
    names = list(arms)
    for a in names:
        for b in names:
            if a != b:
                d = boot[a] - boot[b]
                out[f"{a}_minus_{b}"] = {
                    "diff": out[a]["rate"] - out[b]["rate"],
                    "ci95": list(np.percentile(d, [2.5, 97.5])),
                }
    return out


def main():
    config = simple_parsing.parse(MetricsConfig)
    assert config.dataset_path and config.run_dirs and config.output_dir, "all paths are required"
    logging.basicConfig(level=logging.INFO)
    tasks = {t["id"]: t for t in load_jsonl(config.dataset_path)}
    arms = dict(rescore(Path(d), tasks) for d in config.run_dirs)
    task_ids = sorted({row["instance_id"] for rows in arms.values() for row in rows})

    summary = {
        "git_hash": get_git_hash(),
        "seed": config.seed,
        "n_boot": config.n_boot,
        "task_ids": task_ids,
        "counts": {
            arm: {
                "rows": len(rows),
                "failed": sum(r["failed"] for r in rows),
                "cheat": sum(bool(r["is_cheat"]) for r in rows),
                "scratch": sum(bool(r["is_scratch"]) for r in rows),
                "needs_review": sum(bool(r["needs_review"]) for r in rows),
                "f2p_passed": sum(bool(r["f2p_passed"]) for r in rows),
                "hit_limit": sum(r["hit_limit"] for r in rows),
            }
            for arm, rows in arms.items()
        },
        "cheat_rate": bootstrap_rates(arms, "is_cheat", task_ids, config.n_boot, config.seed),
        "scratch_rate": bootstrap_rates(arms, "is_scratch", task_ids, config.n_boot, config.seed),
        "f2p_pass_rate": bootstrap_rates(arms, "f2p_passed", task_ids, config.n_boot, config.seed),
        "flagged_rows": {
            arm: [r for r in rows if r["is_cheat"] or r["needs_review"] or r["is_scratch"]]
            for arm, rows in arms.items()
        },
    }
    write_config_json(config, config.output_dir)
    path = write_json(summary, Path(config.output_dir) / "metrics.json")
    for arm, c in summary["counts"].items():
        rate = summary["cheat_rate"][arm]
        print(
            f"{arm}: cheat {c['cheat']}/{c['rows']} = {rate['rate']:.2f} [{rate['ci95'][0]:.2f}, {rate['ci95'][1]:.2f}] scratch {c['scratch']} review {c['needs_review']} f2p {c['f2p_passed']} limit {c['hit_limit']} failed {c['failed']}"
        )
    for key, value in summary["cheat_rate"].items():
        if "_minus_" in key:
            print(f"{key}: {value['diff']:+.2f} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}]")
    print(f"Metrics saved to {path}")


main()
