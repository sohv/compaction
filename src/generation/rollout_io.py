# model construction, per-rollout rows and run summaries shared by the rollout and continuation entry points.
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model


def openrouter_model(
    model_id: str,
    provider: str,
    reasoning_enabled: bool | None,
    temperature: float | None,
    max_tokens: int | None,
    seed: int,
) -> Model:
    model_args = {}
    if provider:
        model_args["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning_enabled is not None:
        model_args["reasoning_enabled"] = reasoning_enabled
    generate_config = GenerateConfig(seed=seed, temperature=temperature, max_tokens=max_tokens)
    return get_model(model_id, config=generate_config, **model_args)


def rollout_rows(log: EvalLog, split: str, arm: str) -> list[dict]:
    rows = []
    for sample in log.samples:
        score = next(iter(sample.scores.values())) if sample.scores else None
        rows.append(
            {
                "id": f"{sample.id}__epoch{sample.epoch}",
                "instance_id": sample.id,
                "epoch": sample.epoch,
                "split": split,
                "arm": arm,
                "is_cheat": score.value["is_cheat"] if score else None,
                "f2p_passed": score.value["f2p_passed"] if score else None,
                "is_scratch": bool(score.metadata["diff_check"]["scratch"]) if score else None,
                "needs_review": score.metadata["diff_check"].get("needs_review") if score else None,
                "diff_check": score.metadata["diff_check"] if score else None,
                "p2p_failed": score.metadata["p2p_failed"] if score else None,
                "diff": score.answer if score else None,
                "n_messages": len(sample.messages),
                "hit_limit": sample.limit.type if sample.limit else None,
                "error": sample.error.message if sample.error else None,
            }
        )
    return rows


def run_summary(rows: list[dict], n_expected: int, log: EvalLog) -> dict:
    # errored samples stay as rows with null scores, a dropped row is a silently shrunk dataset
    scored = [row for row in rows if row["is_cheat"] is not None]
    return {
        "n_expected": n_expected,
        "n_rows": len(rows),
        "n_failed": len(rows) - len(scored),
        "cheat_rate": sum(row["is_cheat"] for row in scored) / len(scored) if scored else None,
        "scratch_rate": sum(row["is_scratch"] for row in scored) / len(scored) if scored else None,
        "f2p_pass_rate": sum(row["f2p_passed"] for row in scored) / len(scored) if scored else None,
        # on the conflicting split a passing contradicting test means special-casing, so the headline is the union
        "cheat_or_pass_rate": sum(row["is_cheat"] or row["f2p_passed"] for row in scored) / len(scored)
        if scored
        else None,
        "n_hit_limit": sum(row["hit_limit"] is not None for row in rows),
        "inspect_log": log.location,
    }
