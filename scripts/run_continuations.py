# continues each on-policy transcript from its cut in one arm (full T, S(T), or S(T) with a planted skip), scoring every rollout.
# uv run -m scripts.run_continuations --dataset_path data/processed/tasks.jsonl --cuts_path results/raw/261021_cuts_k3_v3/cuts.jsonl --output_dir results/raw/261021_exp23_k3_full_v1 --model_id openrouter/moonshotai/kimi-k3 --provider moonshotai/mxfp4 --arm full --n_repeats 5 --num_tasks 10 --reasoning_enabled true --temperature 1.0 --max_tokens 8192 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser

from src.data.io import load_jsonl
from src.generation.continuations import (
    ARMS,
    continuation_messages,
    continuation_task,
    cut_call,
    replay_calls,
)
from src.generation.rollout_io import openrouter_model, rollout_rows, run_summary
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class ContinuationConfig(Config):
    cuts_path: str = ""  # from scripts.build_cuts
    arm: str = "full"
    split: str = "conflicting"
    max_sandboxes: int = 8
    max_connections: int = 10
    provider: str = ""
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    budget: int = 0  # new messages per rollout; 0 keeps each cut's own budget
    summary_field: str = "summary"  # which cuts.jsonl field holds this arm's S(T), e.g. an edited copy
    closing: str = ""  # a user message appended to every context, empty for none
    expect_at_cut: str = "fail"  # what the replay check expects of the target test at the cut


def main():
    config = simple_parsing.parse(ContinuationConfig, add_config_path_arg=True)
    assert config.dataset_path and config.cuts_path, "--dataset_path and --cuts_path are required"
    assert config.arm in ARMS, f"unknown arm {config.arm}, choose from {ARMS}"
    assert config.expect_at_cut in ("fail", "pass"), f"unknown expect_at_cut {config.expect_at_cut}"
    load_dotenv()
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    tasks = load_jsonl(config.dataset_path, config.num_tasks)
    cuts = {c["id"]: c for c in load_jsonl(config.cuts_path)}
    assert all(t["id"] in cuts for t in tasks), f"cuts missing for some tasks in {config.cuts_path}"
    source_logs = {path: read_eval_log(path) for path in {cuts[t["id"]]["source_log"] for t in tasks}}
    transcripts = {
        t["id"]: next(
            s.messages
            for s in source_logs[cuts[t["id"]]["source_log"]].samples
            if s.id == t["id"] and s.epoch == cuts[t["id"]]["epoch"]
        )
        for t in tasks
    }

    messages_by_id, replays_by_id, budgets_by_id = {}, {}, {}
    for t in tasks:
        cut, messages = cuts[t["id"]], transcripts[t["id"]]
        context = continuation_messages(config.arm, messages, cut["cut"], cut[config.summary_field])
        messages_by_id[t["id"]] = [*context, ChatMessageUser(content=config.closing)] if config.closing else context
        replays_by_id[t["id"]] = {
            "calls": replay_calls(messages, cut["cut"]),
            "cut_call": cut_call(messages, cut["cut"]),
            "test_name": cut["test_name"],
            "expect": config.expect_at_cut,
        }
        budgets_by_id[t["id"]] = config.budget or cut["budget"]
        LOGGER.info(
            f"{t['id']}: {len(replays_by_id[t['id']]['calls'])} calls to replay, budget {budgets_by_id[t['id']]}"
        )

    task = continuation_task(
        tasks,
        config.split,
        messages_by_id,
        replays_by_id,
        budgets_by_id,
        Path("cache/compose"),
        output_dir / "generations" / "test_logs",
        config.n_repeats,
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
