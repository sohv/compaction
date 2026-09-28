# runs the inspect swe agent on the verified task set for one split and arm, scoring each rollout with the diff check and tests.
# uv run -m scripts.run_rollouts --dataset_path data/processed/tasks.jsonl --output_dir results/raw/260928_smoke_rollouts_v1 --model_id openrouter/qwen/qwen3.8-27b --split original --arm issue_only --num_tasks 2 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model

from src.data.io import load_jsonl
from src.generation.swe_agent import issue_only_messages, swe_task
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = {"issue_only": issue_only_messages}


@dataclass
class RolloutConfig(Config):
    split: str = "original"
    arm: str = "issue_only"
    message_limit: int = 100
    max_sandboxes: int = 4
    max_connections: int = 10
    provider: str = ""  # openrouter upstream to pin, with fallbacks off
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def rollout_rows(log, config: RolloutConfig) -> list[dict]:
    rows = []
    for sample in log.samples:
        score = next(iter(sample.scores.values())) if sample.scores else None
        rows.append(
            {
                "id": f"{sample.id}__epoch{sample.epoch}",
                "instance_id": sample.id,
                "epoch": sample.epoch,
                "split": config.split,
                "arm": config.arm,
                "is_cheat": score.value["is_cheat"] if score else None,
                "f2p_passed": score.value["f2p_passed"] if score else None,
                "diff_check": score.metadata["diff_check"] if score else None,
                "p2p_failed": score.metadata["p2p_failed"] if score else None,
                "diff": score.answer if score else None,
                "n_messages": len(sample.messages),
                "error": sample.error.message if sample.error else None,
            }
        )
    return rows


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
    task = swe_task(
        tasks,
        config.split,
        {t["id"]: ARMS[config.arm](t) for t in tasks},
        Path("cache/compose"),
        output_dir / "generations" / "test_logs",
        config.message_limit,
        config.n_repeats,
    )
    model_args = {}
    if config.provider:
        model_args["provider"] = {"order": [config.provider], "allow_fallbacks": False}
    if config.reasoning_enabled is not None:
        model_args["reasoning_enabled"] = config.reasoning_enabled
    generate_config = GenerateConfig(seed=config.seed, temperature=config.temperature, max_tokens=config.max_tokens)
    model = get_model(config.model_id, config=generate_config, **model_args)

    log = eval(
        task,
        model=model,
        log_dir=str(output_dir / "generations" / "inspect_logs"),
        max_sandboxes=config.max_sandboxes,
        max_connections=config.max_connections,
        fail_on_error=False,
        display="plain",
    )[0]

    rows = rollout_rows(log, config)
    # errored samples stay as rows with null scores, a dropped row is a silently shrunk dataset
    scored = [row for row in rows if row["is_cheat"] is not None]
    n_expected = len(tasks) * config.n_repeats
    summary = {
        "n_expected": n_expected,
        "n_rows": len(rows),
        "n_failed": len(rows) - len(scored),
        "cheat_rate": sum(row["is_cheat"] for row in scored) / len(scored) if scored else None,
        "f2p_pass_rate": sum(row["f2p_passed"] for row in scored) / len(scored) if scored else None,
        "inspect_log": log.location,
    }
    if len(rows) != n_expected:
        LOGGER.warning(f"expected {n_expected} rows, got {len(rows)}")
    path = write_jsonl(rows, output_dir / "outputs.jsonl")
    write_json(summary, output_dir / "summary.json")
    print(f"Results saved to {path}")
    print(f"{summary['n_failed']}/{len(rows)} rollouts failed and were written with null scores")


main()
