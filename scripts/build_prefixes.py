# captures the real repo outputs (listing, failing test source, test failure) that the fabricated pre-compaction turns replay.
# uv run -m scripts.build_prefixes --dataset_path data/processed/tasks.jsonl --split original --output_path data/processed/prefixes_original.jsonl

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import docker
import simple_parsing

from src.data.io import load_jsonl
from src.generation.arms import prefix_commands
from src.generation.swe_agent import SETUP_SCRIPT
from src.metrics.io import write_jsonl
from src.metrics.test_runner import capture_commands

LOGGER = logging.getLogger(__name__)


@dataclass
class PrefixConfig:
    dataset_path: str = ""
    output_path: str = ""
    split: str = "original"
    max_workers: int = 3
    timeout: int = 1800


def capture(client: docker.DockerClient, task: dict, split: str, timeout: int) -> dict:
    setup = SETUP_SCRIPT.format(base_commit=task["base_commit"])
    test_patch = task[f"{split}_test_patch"]
    listing_cmd, grep_cmd = prefix_commands(task)[:2]
    listing, grep_out = capture_commands(client, task["image"], test_patch, setup, [listing_cmd, grep_cmd], timeout)
    assert grep_out.strip(), f"{task['id']}: test function not found with {grep_cmd!r}"
    line = int(grep_out.strip().splitlines()[0].split(":")[0])
    view_cmd, test_cmd = prefix_commands(task, line)[2:]
    view, test_out = capture_commands(client, task["image"], test_patch, setup, [view_cmd, test_cmd], timeout)
    LOGGER.info(f"{task['id']}: captured {len(listing)}+{len(view)}+{len(test_out)} chars")
    return {
        "id": task["id"],
        "split": split,
        "turns": [
            {"command": listing_cmd, "output": listing},
            {"command": grep_cmd, "output": grep_out},
            {"command": view_cmd, "output": view},
            {"command": test_cmd, "output": test_out},
        ],
    }


def main():
    config = simple_parsing.parse(PrefixConfig)
    assert config.dataset_path and config.output_path, "--dataset_path and --output_path are required"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tasks = load_jsonl(config.dataset_path)
    client = docker.from_env(timeout=config.timeout)
    with ThreadPoolExecutor(config.max_workers) as pool:
        rows = list(pool.map(lambda task: capture(client, task, config.split, config.timeout), tasks))
    path = write_jsonl(rows, config.output_path)
    print(f"Prefixes saved to {path}")


main()
