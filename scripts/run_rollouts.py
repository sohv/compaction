# runs the inspect swe agent on the verified task set for one split and arm, scoring each rollout with the diff check and tests.
# uv run -m scripts.run_rollouts --dataset_path data/processed/tasks.jsonl --output_dir results/raw/260928_smoke_rollouts_v1 --model_id openrouter/qwen/qwen3.8-27b --split original --arm issue_only --num_tasks 2 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from inspect_ai import eval

from src.data.io import load_jsonl
from src.generation.arms import ARMS, NO_PREFIX_ARMS
from src.generation.rollout_io import openrouter_model, rollout_rows, run_summary
from src.generation.swe_agent import swe_task
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RolloutConfig(Config):
    split: str = "original"
    arm: str = "issue_only"
    prefixes_path: str = ""  # from scripts.build_prefixes, required for every arm outside NO_PREFIX_ARMS
    message_limit: int = 100
    prohibit_test_edits: bool = True  # the "DO NOT MODIFY THE TESTS" line in the system prompt
    max_attempts: int = 1  # submissions per rollout, above 1 only for the feedback robustness check
    max_sandboxes: int = 4
    max_connections: int = 10
    provider: str = ""  # openrouter upstream to pin, with fallbacks off
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def main():
    config = simple_parsing.parse(RolloutConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"
    assert config.split in ("original", "conflicting"), f"unknown split {config.split}"
    load_dotenv()

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    config_path = write_config_json(config, output_dir)
    LOGGER.info(f"wrote run metadata to {config_path}")

    tasks = load_jsonl(config.dataset_path, config.num_tasks)
    LOGGER.info(f"loaded {len(tasks)} tasks from {config.dataset_path}")
    assert config.arm in ARMS, f"unknown arm {config.arm}, choose from {sorted(ARMS)}"
    prefixes = {}
    if config.arm not in NO_PREFIX_ARMS:
        assert config.prefixes_path, f"--prefixes_path is required for arm {config.arm}"
        prefixes = {p["id"]: p for p in load_jsonl(config.prefixes_path) if p["split"] == config.split}
        assert all(t["id"] in prefixes for t in tasks), f"prefixes missing for some tasks in {config.prefixes_path}"
    task = swe_task(
        tasks,
        config.split,
        {t["id"]: ARMS[config.arm](t, prefixes.get(t["id"])) for t in tasks},
        Path("cache/compose"),
        output_dir / "generations" / "test_logs",
        config.message_limit,
        config.n_repeats,
        config.prohibit_test_edits,
        config.max_attempts,
    )
    model = openrouter_model(
        config.model_id, config.provider, config.reasoning_enabled, config.temperature, config.max_tokens, config.seed
    )

    log = eval(
        task,
        model=model,
        log_dir=str(output_dir / "generations" / "inspect_logs"),
        max_sandboxes=config.max_sandboxes,
        max_connections=config.max_connections,
        fail_on_error=False,
        display="plain",
    )[0]

    rows = rollout_rows(log, config.split, config.arm)
    n_expected = len(tasks) * config.n_repeats
    summary = run_summary(rows, n_expected, log)
    if len(rows) != n_expected:
        LOGGER.warning(f"expected {n_expected} rows, got {len(rows)}")
    path = write_jsonl(rows, output_dir / "outputs.jsonl")
    write_json(summary, output_dir / "summary.json")
    print(f"Results saved to {path}")
    print(f"{summary['n_failed']}/{len(rows)} rollouts failed and were written with null scores")


main()
