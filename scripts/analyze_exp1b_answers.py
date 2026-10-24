# post hoc rescoring of experiment 1b's direct runs: the first line anywhere in the continuation where Yes or No stands alone.
# uv run -m scripts.analyze_exp1b_answers --run_root results/raw --output_dir results/raw/261024_exp1b_posthoc_answers_v1 --model_id openrouter/moonshotai/kimi-k3 --seed 42

import glob
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from inspect_ai.log import read_eval_log

from src.generation.disclosure import STANDALONE_ANSWER, standalone_answer
from src.metrics.io import write_json, write_jsonl
from src.metrics.stats import cluster_bootstrap_diff, cluster_bootstrap_rate, paired_permutation_p
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
ARMS = ("full_noreason", "summary")
BUDGET = 30


@dataclass
class AnswersConfig(Config):
    run_root: str = "results/raw"


def assistant_texts(new: list) -> list[str]:
    texts = []
    for m in new:
        if m.role == "assistant":
            texts.append(m.text)
            texts += [str(tc.arguments.get("answer", "")) for tc in m.tool_calls or [] if tc.function == "submit"]
    return texts


def run_rows(run_dir: Path, arm: str) -> list[dict]:
    logs = glob.glob(str(run_dir / "generations" / "inspect_logs" / "*.eval"))
    assert len(logs) == 1, f"expected exactly one inspect log in {run_dir}, found {len(logs)}"
    rows = []
    for s in read_eval_log(logs[0]).samples:
        new = s.messages[len(s.input) + 1 :]
        texts = assistant_texts(new)
        answer = standalone_answer(texts)
        line = next((m.group(0).strip() for t in texts for m in [STANDALONE_ANSWER.search(t)] if m), None)
        rows.append(
            {
                "id": f"{s.id}__epoch{s.epoch}__{arm}__direct",
                "transcript_id": s.id,
                "task_id": s.id.split("__", 1)[1].rsplit("__epoch", 1)[0],
                "arm": arm,
                "is_failed": s.error is not None,
                "answer": answer,
                "answer_line": line,
                "said_yes": int(answer == "yes"),
                "said_no": int(answer == "no"),
                "answered": int(answer is not None),
                "n_new_messages": len(new),
                "submitted": any(
                    tc.function == "submit" for m in new if m.role == "assistant" for tc in m.tool_calls or []
                ),
            }
        )
    return rows


def main():
    config = simple_parsing.parse(AnswersConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    root = Path(config.run_root)

    runs = {a: run_rows(root / f"261023_exp1b_k3_{a}_direct_v1", a) for a in ARMS}
    failed = {a: sum(r["is_failed"] for r in rows) for a, rows in runs.items()}
    runs = {a: [r for r in rows if not r["is_failed"]] for a, rows in runs.items()}
    answered = {a: [r for r in rows if r["answered"]] for a, rows in runs.items()}

    def compare(a: list[dict], b: list[dict], key: str) -> dict:
        return {
            **cluster_bootstrap_diff(a, b, "transcript_id", key, seed=config.seed),
            "permutation_p": paired_permutation_p(a, b, "transcript_id", key),
        }

    result = {
        "post_hoc": True,
        "rates": {
            a: {
                k: {
                    "k": sum(r[k] for r in rows),
                    "n": len(rows),
                    **cluster_bootstrap_rate(rows, "transcript_id", k, seed=config.seed),
                }
                for k in ("said_yes", "said_no", "answered")
            }
            for a, rows in runs.items()
        },
        "yes_full_minus_summary": compare(runs["full_noreason"], runs["summary"], "said_yes"),
        "yes_among_answered_full_minus_summary": compare(answered["full_noreason"], answered["summary"], "said_yes"),
        "per_transcript_yes": {
            t: {a: sum(r["said_yes"] for r in runs[a] if r["transcript_id"] == t) for a in ARMS}
            for t in sorted({r["transcript_id"] for a in ARMS for r in runs[a]})
        },
        "unanswered_at_budget": {
            a: sum(1 for r in runs[a] if not r["answered"] and r["n_new_messages"] >= BUDGET - 1) for a in ARMS
        },
        "n_failed": failed,
    }
    write_jsonl([r for rows in runs.values() for r in rows], output_dir / "rows.jsonl")
    path = write_json(result, output_dir / "analysis.json")
    for a, r in result["rates"].items():
        print(a, {k: f"{v['k']}/{v['n']}" for k, v in r.items()})
    for k in ("yes_full_minus_summary", "yes_among_answered_full_minus_summary"):
        c = result[k]
        print(f"{k}: {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] perm {c['permutation_p']:.4f}")
    print("unanswered at budget:", result["unanswered_at_budget"], "| failed:", failed)
    print(f"Results saved to {path}")


main()
