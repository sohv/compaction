# verifies gold patches for the ranked candidates in docker and writes the first 30 that pass as the task set.
# uv run -m scripts.verify_tasks --dataset_path data/raw/impossible_swebench --verified_path data/raw/swebench_verified/test.parquet --output_dir data/splits --seed 42

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import docker
import pandas as pd
import simple_parsing

from src.data.impossible_swebench import load_split
from src.metrics.io import write_json, write_jsonl
from src.metrics.test_runner import ensure_image, eval_script_for, f2p_status, run_tests

LOGGER = logging.getLogger(__name__)


@dataclass
class VerifyConfig:
    dataset_path: str = ""
    verified_path: str = ""
    candidates_path: str = "data/splits/task_candidates.json"
    output_dir: str = ""
    tasks_path: str = "data/processed/tasks.jsonl"
    n_tasks: int = 30
    max_workers: int = 4
    timeout: int = 1800
    seed: int = 42


def verify(client, row: dict, conflicting_test_patch: str, log_dir: Path, timeout: int) -> dict:
    ensure_image(client, row["image"])
    arms = {
        "original_unpatched": ("", row["test_patch"]),
        "original_gold": (row["patch"], row["test_patch"]),
        "conflicting_gold": (row["patch"], conflicting_test_patch),
    }
    result = {"id": row["instance_id"]}
    for arm, (code_patch, test_patch) in arms.items():
        script = eval_script_for(row, test_patch)
        log, is_complete = run_tests(client, row["image"], code_patch, script, timeout)
        result[arm] = {
            "completed": is_complete,
            **f2p_status(row, script, log, log_dir / f"{row['instance_id']}.{arm}.log"),
        }
    result["passes"] = all(result[arm]["completed"] and result[arm]["log_valid"] for arm in arms) and (
        bool(result["original_unpatched"]["f2p_failed"])
        and not result["original_gold"]["f2p_failed"]
        and bool(result["conflicting_gold"]["f2p_failed"])
    )
    LOGGER.info(f"{row['instance_id']}: passes={result['passes']}")
    return result


def verify_or_record(client, row: dict, conflicting_test_patch: str, log_dir: Path, timeout: int) -> dict:
    try:
        return verify(client, row, conflicting_test_patch, log_dir, timeout)
    except Exception as error:
        LOGGER.exception(f"verification crashed for {row['instance_id']}")
        return {"id": row["instance_id"], "passes": False, "error": repr(error)}


def main():
    config = simple_parsing.parse(VerifyConfig)
    assert config.dataset_path and config.verified_path and config.output_dir, "all input and output paths are required"
    output_dir = Path(config.output_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    original = load_split(config.dataset_path, "original")
    conflicting = load_split(config.dataset_path, "conflicting")
    verified = pd.read_parquet(config.verified_path).set_index("instance_id", drop=False)
    candidate_ids = [
        c["instance_id"] for c in json.loads(Path(config.candidates_path).read_text())["ranked_candidates"]
    ]

    client = docker.from_env(timeout=config.timeout)
    with ThreadPoolExecutor(config.max_workers) as pool:
        results = list(
            pool.map(
                lambda iid: verify_or_record(
                    client,
                    verified.loc[iid].to_dict(),
                    conflicting.loc[iid, "test_patch"],
                    output_dir / "verification_logs",
                    config.timeout,
                ),
                candidate_ids,
            )
        )

    selected = [r["id"] for r in results if r["passes"]][: config.n_tasks]
    assert (
        len(selected) == config.n_tasks
    ), f"only {len(selected)} candidates passed, extend --n_candidates in select_tasks"
    verification_path = write_json(
        {"selected": selected, "n_passed": sum(r["passes"] for r in results), "results": results},
        output_dir / "task_verification.json",
    )
    tasks = [
        {
            "id": iid,
            **{
                k: original.loc[iid, k]
                for k in ("repo", "base_commit", "problem_statement", "first_fail_to_pass_test", "patch")
            },
            "FAIL_TO_PASS": list(original.loc[iid, "FAIL_TO_PASS"]),
            "original_test_patch": original.loc[iid, "test_patch"],
            "conflicting_test_patch": conflicting.loc[iid, "test_patch"],
            **{k: verified.loc[iid, k] for k in ("image", "log_parser", "eval_type", "eval_script", "version")},
            "PASS_TO_PASS": list(verified.loc[iid, "PASS_TO_PASS"]),
        }
        for iid in selected
    ]
    tasks_path = write_jsonl(tasks, config.tasks_path)
    print(f"Verification saved to {verification_path}")
    print(f"Task set saved to {tasks_path}")
    print(f"{len(selected)} tasks selected, {verification_path} lists every candidate's outcome")


main()
