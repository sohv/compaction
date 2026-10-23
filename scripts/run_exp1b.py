# experiment 1b: continues each real cheat, with tools, from its reasoning-free transcript or its summary, ending in a report request or a direct question.
# uv run -m scripts.run_exp1b --dataset_path data/processed/tasks.jsonl --cuts_path results/raw/261022_exp1_k3_v2/cuts.jsonl --output_dir results/raw/261023_exp1b_k3_full_noreason_direct_v1 --arm full_noreason --prompt direct --model_id openrouter/moonshotai/kimi-k3 --provider moonshotai/mxfp4 --n_repeats 5 --reasoning_enabled true --temperature 1.0 --max_tokens 8192 --seed 42

import glob
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
    continuation_messages,
    continuation_task,
    cut_call,
    replay_calls,
    target_test_name,
)
from src.generation.disclosure import CLEAN_DIRECT_QUESTION, CLEAN_REPORT_REQUEST
from src.generation.rollout_io import openrouter_model, rollout_rows, run_summary
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PROMPTS = {"report": CLEAN_REPORT_REQUEST, "direct": CLEAN_DIRECT_QUESTION}
ARMS = ("full_noreason", "summary")


@dataclass
class Exp1bConfig(Config):
    cuts_path: str = ""  # experiment 1's cuts: one row per cheated transcript, cut where the target test first passes
    arm: str = "full_noreason"
    prompt: str = "direct"
    budget: int = 30
    split: str = "conflicting"
    max_sandboxes: int = 8
    max_connections: int = 10
    provider: str = ""
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def source_messages(source: str, task_id: str, epoch: int) -> list:
    logs = glob.glob(f"{source}/generations/inspect_logs/*.eval")
    assert len(logs) == 1, f"expected exactly one inspect log in {source}, found {len(logs)}"
    return next(s.messages for s in read_eval_log(logs[0]).samples if s.id == task_id and s.epoch == epoch)


def main():
    config = simple_parsing.parse(Exp1bConfig, add_config_path_arg=True)
    assert config.dataset_path and config.cuts_path, "--dataset_path and --cuts_path are required"
    assert config.arm in ARMS, f"unknown arm {config.arm}, choose from {ARMS}"
    assert config.prompt in PROMPTS, f"unknown prompt {config.prompt}, choose from {sorted(PROMPTS)}"
    load_dotenv()
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    tasks_by_id = {t["id"]: t for t in load_jsonl(config.dataset_path)}
    cuts = load_jsonl(config.cuts_path)
    # one sample per transcript: seven transcripts share sympy-22914, so the sample id is the transcript id
    tasks, messages_by_id, replays_by_id, budgets_by_id = [], {}, {}, {}
    for c in cuts:
        task = {**tasks_by_id[c["task_id"]], "id": c["id"]}
        messages = source_messages(c["source"], c["task_id"], c["epoch"])
        context = continuation_messages(config.arm, messages, c["cut"], c["summary"])
        tasks.append(task)
        messages_by_id[c["id"]] = [*context, ChatMessageUser(content=PROMPTS[config.prompt])]
        replays_by_id[c["id"]] = {
            "calls": replay_calls(messages, c["cut"]),
            "cut_call": cut_call(messages, c["cut"]),
            "test_name": target_test_name(task),
            "expect": "pass",
        }
        budgets_by_id[c["id"]] = config.budget
        LOGGER.info(
            f"{c['id']}: {len(replays_by_id[c['id']]['calls'])} calls to replay, context {len(context)} messages"
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
    rows = rollout_rows(log, config.split, f"{config.arm}_{config.prompt}")
    summary = run_summary(rows, len(tasks) * config.n_repeats, log)
    path = write_jsonl(rows, output_dir / "outputs.jsonl")
    write_json(summary, output_dir / "summary.json")
    print(f"Results saved to {path}")
    print(f"{summary['n_failed']}/{len(rows)} rollouts failed and were written with null scores")


main()
