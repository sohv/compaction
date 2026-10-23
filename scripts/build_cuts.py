# picks the cut in one experiment 0 transcript per task and has the agent model write S(T) with codex's compaction prompt.
# uv run -m scripts.build_cuts --dataset_path data/processed/tasks.jsonl --source_log results/raw/261021_exp0_k3_issue_only_v1/generations/inspect_logs/2026-10-21T05-20-57-00-00_task_aLhFuvkRF6Dr6NjXMu25aD.eval --output_dir results/raw/261021_cuts_k3_v3 --num_tasks 10 --model_id openrouter/moonshotai/kimi-k3 --provider moonshotai/mxfp4 --reasoning_enabled true --temperature 1.0 --max_tokens 8192 --seed 42

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

from src.data.io import load_jsonl
from src.generation.continuations import (
    cut_call,
    find_cut,
    first_cut,
    message_budget,
    target_test_name,
)
from src.generation.rollout_io import openrouter_model
from src.generation.summaries import summarize_one
from src.metrics.constraint import git_violations
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class CutsConfig(Config):
    source_log: str = ""  # the experiment 0 issue_only inspect log the transcripts come from
    epoch: int = 1
    git_compliant: bool = False  # experiment 4c: take the first epoch up to max_epoch that ran no git before the cut
    max_epoch: int = 2
    provider: str = ""
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    summary_max_tokens: int = (
        32768  # reasoning on a long transcript can exhaust the 8192 agent-turn cap before any text
    )


async def summarize(model, samples: list, cuts: list[int]) -> list:
    return await asyncio.gather(*[summarize_one(model, s.messages, cut) for s, cut in zip(samples, cuts)])


def select_git_compliant(log, tasks: dict, max_epoch: int) -> tuple[list, dict]:
    """Per task, the first epoch with a cut and no git call before it; a task with none is dropped and reported."""
    samples, selection = [], {}
    for task_id, task in tasks.items():
        epochs = {s.epoch: s for s in log.samples if s.id == task_id}
        report = {}
        for epoch in range(1, max_epoch + 1):
            cut = first_cut(epochs[epoch].messages, task)
            violation = git_violations(epochs[epoch].messages[: cut + 1]) if cut is not None else None
            report[epoch] = {"cut": cut, "git_before_cut": violation["commands"] if violation else None}
            if cut is not None and not violation["violated"]:
                samples.append(epochs[epoch])
                report["chosen_epoch"] = epoch
                break
        selection[task_id] = report
    return samples, selection


def main():
    config = simple_parsing.parse(CutsConfig, add_config_path_arg=True)
    assert config.dataset_path and config.source_log, "--dataset_path and --source_log are required"
    load_dotenv()
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    tasks = {t["id"]: t for t in load_jsonl(config.dataset_path, config.num_tasks)}
    log = read_eval_log(config.source_log)
    if config.git_compliant:
        samples, selection = select_git_compliant(log, tasks, config.max_epoch)
        write_json(selection, output_dir / "selection.json")
        LOGGER.info(f"{len(samples)} of {len(tasks)} tasks have a transcript with no git before the cut")
    else:
        samples = sorted(
            [s for s in log.samples if s.epoch == config.epoch and s.id in tasks], key=lambda s: list(tasks).index(s.id)
        )
        assert len(samples) == len(tasks), f"expected one epoch {config.epoch} transcript per task, got {len(samples)}"
    cuts = [find_cut(s.messages, tasks[s.id]) for s in samples]

    model = openrouter_model(
        config.model_id,
        config.provider,
        config.reasoning_enabled,
        config.temperature,
        config.summary_max_tokens,
        config.seed,
    )
    outputs = asyncio.run(summarize(model, samples, cuts))

    rows = []
    for s, cut, (output, attempts) in zip(samples, cuts, outputs):
        call = cut_call(s.messages, cut)
        rows.append(
            {
                "id": s.id,
                "epoch": s.epoch,
                "source_log": config.source_log,
                "cut": cut,
                "budget": message_budget(cut),
                "test_name": target_test_name(tasks[s.id]),
                "cut_call": call.arguments,
                "cut_output_tail": s.messages[cut].text[-1500:],
                "summary": output.completion,
                "summary_stop_reason": output.stop_reason,
                "summary_attempts": attempts,
                "summary_tokens": output.usage.output_tokens if output.usage else None,
            }
        )
        LOGGER.info(f"{s.id}: cut at message {cut} of {len(s.messages)}, budget {message_budget(cut)}")
    path = write_jsonl(rows, output_dir / "cuts.jsonl")
    print(f"Results saved to {path}")


main()
