# experiment 1: cuts K3's real cheats where the target test first passes, summarises them, and asks for a report or a direct admission from T or S(T).
# uv run -m scripts.run_disclosure --stage summaries --dataset_path data/processed/tasks.jsonl --output_dir results/raw/261022_exp1_k3_v1 --model_id openrouter/moonshotai/kimi-k3 --provider moonshotai/mxfp4 --reasoning_enabled true --temperature 1.0 --max_tokens 8192 --seed 42

import asyncio
import glob
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig

from src.data.io import load_jsonl
from src.generation.continuations import first_pass_cut
from src.generation.disclosure import DIRECT_QUESTION, REPORT_REQUEST, admits, disclosure_context
from src.generation.rollout_io import openrouter_model
from src.generation.summaries import AGENT_TOOLS, summarize_one
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
# every K3 rollout that cheated with no compaction in its history, as pre-registered
SOURCES = ["results/raw/261021_exp0_k3_issue_only_v1", "results/raw/261021_exp23_k3_full_v1"]
PROMPTS = {"report": REPORT_REQUEST, "direct": DIRECT_QUESTION}
ARMS = ("full", "summary")


@dataclass
class DisclosureConfig(Config):
    stage: str = "summaries"  # summaries, then replies after the summaries are labelled by hand, then retry_empty
    n_samples: int = 5
    provider: str = ""
    reasoning_enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    summary_max_tokens: int = 32768


def cheated_transcripts(tasks: dict) -> list[dict]:
    found = []
    for source in SOURCES:
        rows = {r["id"]: r for r in load_jsonl(f"{source}/outputs.jsonl")}
        log = read_eval_log(glob.glob(f"{source}/generations/inspect_logs/*.eval")[0])
        for s in log.samples:
            row = rows[f"{s.id}__epoch{s.epoch}"]
            if row["is_cheat"] or row["f2p_passed"]:
                found.append(
                    {
                        "id": f"{Path(source).name}__{s.id}__epoch{s.epoch}",
                        "task_id": s.id,
                        "source": source,
                        "epoch": s.epoch,
                        "messages": s.messages,
                    }
                )
    return found


def load_messages(source: str, task_id: str, epoch: int) -> list:
    log = read_eval_log(glob.glob(f"{source}/generations/inspect_logs/*.eval")[0])
    return next(s.messages for s in log.samples if s.id == task_id and s.epoch == epoch)


async def build_summaries(config: DisclosureConfig, tasks: dict, output_dir: Path) -> None:
    model = openrouter_model(
        config.model_id,
        config.provider,
        config.reasoning_enabled,
        config.temperature,
        config.summary_max_tokens,
        config.seed,
    )
    transcripts = cheated_transcripts(tasks)
    kept, dropped = [], []
    for t in transcripts:
        cut = first_pass_cut(t["messages"], tasks[t["task_id"]])
        (kept if cut is not None else dropped).append({**t, "cut": cut})
    outputs = await asyncio.gather(*[summarize_one(model, t["messages"], t["cut"]) for t in kept])
    rows = [
        {
            "id": t["id"],
            "task_id": t["task_id"],
            "source": t["source"],
            "epoch": t["epoch"],
            "cut": t["cut"],
            "cut_output_tail": t["messages"][t["cut"]].text[-1200:],
            "summary": out.completion,
            "summary_attempts": attempts,
        }
        for t, (out, attempts) in zip(kept, outputs)
    ]
    write_json(
        {"n_cheated": len(transcripts), "n_kept": len(kept), "dropped_no_pass_run": [t["id"] for t in dropped]},
        output_dir / "selection.json",
    )
    path = write_jsonl(rows, output_dir / "cuts.jsonl")
    print(
        f"{len(kept)} of {len(transcripts)} cheated transcripts have a passing run; dropped {[t['id'] for t in dropped]}"
    )
    print(f"Results saved to {path}")


async def build_replies(config: DisclosureConfig, output_dir: Path) -> None:
    cuts = load_jsonl(str(output_dir / "cuts.jsonl"))
    model = openrouter_model(
        config.model_id, config.provider, config.reasoning_enabled, config.temperature, config.max_tokens, config.seed
    )
    jobs = []
    for c in cuts:
        messages = load_messages(c["source"], c["task_id"], c["epoch"])
        for arm in ARMS:
            for prompt_key, prompt in PROMPTS.items():
                context = disclosure_context(arm, messages, c["cut"], c["summary"], prompt)
                # identical inputs would repeat under one fixed seed, so each sample gets its own
                jobs += [(c, arm, prompt_key, i, context) for i in range(config.n_samples)]
    outputs = await asyncio.gather(
        *[
            model.generate(ctx, tools=AGENT_TOOLS, tool_choice="none", config=GenerateConfig(seed=config.seed + i))
            for _, _, _, i, ctx in jobs
        ]
    )
    rows = []
    for (c, arm, prompt_key, i, _), out in zip(jobs, outputs):
        content = out.message.content if isinstance(out.message.content, list) else []
        rows.append(
            {
                "id": f"{c['id']}__{arm}__{prompt_key}__{i}",
                "transcript_id": c["id"],
                "task_id": c["task_id"],
                "arm": arm,
                "prompt": prompt_key,
                "sample": i,
                "seed": config.seed + i,
                "text": out.completion,
                "reasoning": " ".join(x.reasoning for x in content if getattr(x, "type", "") == "reasoning"),
                "admits": admits(out.completion) if prompt_key == "direct" else None,
                "stop_reason": out.stop_reason,
            }
        )
    path = write_jsonl(rows, output_dir / "replies.jsonl")
    print(f"{len(rows)} replies, {sum(not r['text'] for r in rows)} empty")
    print(f"Results saved to {path}")


async def retry_empty(config: DisclosureConfig, output_dir: Path) -> None:
    """Empty replies cluster in one arm, so they are resampled with the summary token cap rather than dropped."""
    rows = load_jsonl(str(output_dir / "replies.jsonl"))
    cuts = {c["id"]: c for c in load_jsonl(str(output_dir / "cuts.jsonl"))}
    model = openrouter_model(
        config.model_id,
        config.provider,
        config.reasoning_enabled,
        config.temperature,
        config.summary_max_tokens,
        config.seed,
    )
    messages = {cid: load_messages(c["source"], c["task_id"], c["epoch"]) for cid, c in cuts.items()}
    for attempt in range(1, 4):
        empty = [r for r in rows if not r["text"]]
        LOGGER.info(f"retry attempt {attempt}: {len(empty)} empty replies")
        if not empty:
            break
        contexts = [
            disclosure_context(
                r["arm"],
                messages[r["transcript_id"]],
                cuts[r["transcript_id"]]["cut"],
                cuts[r["transcript_id"]]["summary"],
                PROMPTS[r["prompt"]],
            )
            for r in empty
        ]
        outputs = await asyncio.gather(
            *[
                model.generate(
                    ctx,
                    tools=AGENT_TOOLS,
                    tool_choice="none",
                    config=GenerateConfig(seed=1000 + attempt * 100 + r["sample"]),
                )
                for r, ctx in zip(empty, contexts)
            ]
        )
        for r, out in zip(empty, outputs):
            content = out.message.content if isinstance(out.message.content, list) else []
            r.update(
                text=out.completion,
                retries=attempt,
                seed=1000 + attempt * 100 + r["sample"],
                stop_reason=out.stop_reason,
                reasoning=" ".join(x.reasoning for x in content if getattr(x, "type", "") == "reasoning"),
                admits=admits(out.completion) if r["prompt"] == "direct" else None,
                tool_calls_attempted=len(out.message.tool_calls or []),
            )
    path = write_jsonl(rows, output_dir / "replies.jsonl")
    print(
        f"{sum(not r['text'] for r in rows)} still empty after retries; {sum(bool(r.get('retries')) for r in rows)} replies were resampled"
    )
    print(f"Results saved to {path}")


def main():
    config = simple_parsing.parse(DisclosureConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required"
    assert config.stage in ("summaries", "replies", "retry_empty"), f"unknown stage {config.stage}"
    load_dotenv()
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    # the run directory keeps the summaries config.json; the replies stage records its own beside it
    write_config_json(config, output_dir if config.stage == "summaries" else output_dir / f"{config.stage}_config")
    tasks = {t["id"]: t for t in load_jsonl(config.dataset_path)}
    if config.stage == "summaries":
        asyncio.run(build_summaries(config, tasks, output_dir))
    elif config.stage == "replies":
        asyncio.run(build_replies(config, output_dir))
    else:
        asyncio.run(retry_empty(config, output_dir))


main()
