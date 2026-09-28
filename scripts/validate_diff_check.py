# validates the cheat diff check on 20 labelled diffs from the candidate tasks and sweeps it over every real dataset patch.
# uv run -m scripts.validate_diff_check --dataset_path data/raw/impossible_swebench --output_dir results/raw/260928_smoke_diff_check_validation_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.impossible_swebench import DATASET_REVISION, load_split
from src.metrics.cheat_fixtures import CHEAT_KINDS, build_cheat
from src.metrics.diff_check import check_diff
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidateConfig:
    dataset_path: str = ""
    candidates_path: str = "data/splits/task_candidates.json"
    output_dir: str = ""
    seed: int = 42


def labelled_set(original, candidate_ids: list[str]) -> list[dict]:
    honest = [
        {
            "id": f"honest_{iid}",
            "instance_id": iid,
            "label_cheat": False,
            "kind": "gold_patch",
            "diff": original.loc[iid, "patch"],
        }
        for iid in candidate_ids[:10]
    ]
    cheats = [
        {
            "id": f"cheat_{kind}_{iid}",
            "instance_id": iid,
            "label_cheat": True,
            "kind": kind,
            "diff": build_cheat(kind, original.loc[iid].to_dict()),
        }
        for kind, iid in zip(CHEAT_KINDS, candidate_ids[10:20])
    ]
    return honest + cheats


def main():
    config = simple_parsing.parse(ValidateConfig)
    assert config.dataset_path and config.output_dir, "--dataset_path and --output_dir are required"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    original = load_split(config.dataset_path, "original")
    conflicting = load_split(config.dataset_path, "conflicting")
    candidate_ids = [
        c["instance_id"] for c in json.loads(Path(config.candidates_path).read_text())["ranked_candidates"]
    ]

    rows = []
    for item in labelled_set(original, candidate_ids):
        task = original.loc[item["instance_id"]]
        check = check_diff(item["diff"], task.test_patch, task.patch)
        rows.append(
            {
                **item,
                "pred_cheat": check.is_cheat,
                "agrees": check.is_cheat == item["label_cheat"],
                "flags": check.to_dict(),
            }
        )
        if not rows[-1]["agrees"]:
            LOGGER.error(f"disagreement on {item['id']}: {check.to_dict()}")

    gold_flagged = [
        iid for iid, row in original.iterrows() if check_diff(row.patch, row.test_patch, row.patch).is_cheat
    ]
    test_patch_missed = [
        f"{split}:{iid}"
        for split, frame in (("original", original), ("conflicting", conflicting))
        for iid, row in frame.iterrows()
        if not check_diff(row.test_patch).is_cheat
    ]
    hardcode_without_gold = [
        iid for iid, row in original.iterrows() if check_diff(row.patch, row.test_patch).hardcoded_expected
    ]

    summary = {
        "dataset_revision": DATASET_REVISION,
        "labels_by": "constructed by claude, not human hand labels on model rollouts",
        "n_labelled": len(rows),
        "n_agree": sum(r["agrees"] for r in rows),
        "disagreements": [r["id"] for r in rows if not r["agrees"]],
        "gold_patches_flagged": {"n": len(gold_flagged), "of": len(original), "ids": gold_flagged},
        "test_patches_missed": {"n": len(test_patch_missed), "of": 2 * len(original), "ids": test_patch_missed},
        "gold_hardcode_flags_without_gold_exclusion": {
            "n": len(hardcode_without_gold),
            "of": len(original),
            "ids": hardcode_without_gold,
        },
    }
    labelled_path = write_jsonl(rows, output_dir / "labelled_diffs.jsonl")
    summary_path = write_json(summary, output_dir / "summary.json")
    print(f"Labelled diffs saved to {labelled_path}")
    print(f"Summary saved to {summary_path}")
    print(
        f"agreement {summary['n_agree']}/{summary['n_labelled']}, gold flagged {len(gold_flagged)}, test patches missed {len(test_patch_missed)}"
    )


main()
