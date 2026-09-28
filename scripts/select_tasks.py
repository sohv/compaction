# ranks eligible impossiblebench swe-bench instances by the pre-registered selection rule in docs/decisions.md.
# uv run -m scripts.select_tasks --dataset_path data/raw/impossible_swebench --output_dir data/splits --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.impossible_swebench import DATASET_REVISION, eligibility, load_split, rank_candidates
from src.metrics.io import write_json
from src.utils.config import Config

LOGGER = logging.getLogger(__name__)


@dataclass
class SelectConfig(Config):
    max_per_repo: int = 8
    n_candidates: int = 40


def main():
    config = simple_parsing.parse(SelectConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"
    logging.basicConfig(level=logging.INFO)

    original = load_split(config.dataset_path, "original")
    conflicting = load_split(config.dataset_path, "conflicting")
    table = eligibility(original, conflicting)
    criteria = [c for c in table.columns if c not in ("instance_id", "repo")]
    for criterion in criteria:
        LOGGER.info(f"{criterion}: {int(table[criterion].sum())}/{len(table)}")
    eligible = table[table[criteria].all(axis=1)]
    LOGGER.info(f"eligible: {len(eligible)}, by repo: {eligible.repo.value_counts().to_dict()}")

    ranked = rank_candidates(eligible, config.seed, config.max_per_repo, config.n_candidates)
    assert len(ranked) == config.n_candidates, f"only {len(ranked)} candidates under the per-repo cap"
    path = write_json(
        {
            "dataset": "fjzzq2002/impossible_swebench",
            "revision": DATASET_REVISION,
            "seed": config.seed,
            "max_per_repo": config.max_per_repo,
            "n_eligible": len(eligible),
            "eligible_by_repo": eligible.repo.value_counts().to_dict(),
            "ranked_candidates": [
                {"rank": i, "instance_id": iid, "repo": original.loc[iid, "repo"]} for i, iid in enumerate(ranked)
            ],
        },
        Path(config.output_dir) / "task_candidates.json",
    )
    print(f"Candidates saved to {path}")


main()
