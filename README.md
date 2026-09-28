# Compaction vector

Does a model follow a cheating decision planted in a compaction summary, and does it trust that channel more than
the same instruction arriving as a user message? Rollouts continue a coding-agent session on ImpossibleBench
SWE-bench tasks after a fabricated compaction block, with cheating scored mechanically from the model's diff. The
pre-registered bars and task-selection rule are in `docs/decisions.md`.

SWE-bench containers need Docker. On Apple Silicon: `brew install colima docker`, then
`colima start --vm-type vz --vz-rosetta --cpu 6 --memory 10 --disk 200` so the x86 images run under Rosetta.

## Setup

```bash
uv sync --extra dev --extra llm
pre-commit install
```

The `llm` extra carries the Anthropic, OpenAI and LiteLLM SDKs. Without it the scripts in
`scripts/` fail on import, so install it up front unless the project calls no APIs at all.

Add inference-stack dependencies (vLLM, transformers, torch, etc.) to the `inference` extra in
`pyproject.toml` as needed, then:

```bash
uv sync --extra dev --extra llm --extra inference
```

Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored; `.env.example` is the
only file in the repo that may show a key, and only as a placeholder.

## Repo structure

See `docs/tooling.md` for the full layout — `tests/test_conventions.py` checks it against the repo,
so it can't go stale. In short: `src/` holds reusable code by pipeline stage, `scripts/` holds thin
entry points that call into it, `data/` and `results/` hold files, `docs/` holds the rules and the
design record, and `research_log.md` at the root records every run. `coding_guide.md`
walks through the same layout with the rules attached.

## Running an experiment

Every run passes `--output_dir`, `--seed`, and `--model_id`. Use `--num_tasks 10` to confirm a
script runs before committing to a full run.

```bash
uv run -m scripts.run_generation \
  --dataset_path data/processed/prompts.jsonl \
  --output_dir results/raw/250612_example_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 10 \
  --seed 42

uv run -m scripts.run_metrics \
  --dataset_path results/raw/250612_example_v1/outputs.jsonl \
  --output_dir results/raw/250612_example_v1 \
  --model_id claude-sonnet-4-6 \
  --seed 42
```

Never invoke scripts with bare `python`. Everything runs through `uv run -m`.

### Configs

The `Config` dataclass in `src/utils/config.py` is authoritative — it defines the fields, types,
and defaults. A YAML in `configs/` seeds those defaults for one experiment:

```bash
uv run -m scripts.run_generation --config_path configs/example_experiment.yaml --num_tasks 10
```

Precedence is **explicit CLI flag > `--config_path` YAML > dataclass default**. `dataset_path` has
no default and must always be given, by flag or by config.

## Conventions

- `results/raw/` is append-only. A rerun writes a new `YYMMDD_description_v1/` subfolder rather
  than overwriting prior results.
- Every run directory gets a `config.json` (git hash, model, seed) and a `run.log` beside its
  results, so any number can be traced back to the commit that produced it.
- Structured results are committed. Bulk artifacts too large for git go in
  `results/raw/<run>/generations/`, which is gitignored — everything else in the run directory
  stays in version control.
- JSON for under 1000 records (summaries, metrics, configs), JSONL at 1000 or above. One JSON
  object per line, `id` first. Floats rounded to 4 decimal places.
- Per-run figures go in `<output_dir>/figures/`. Polished figures for the paper go in
  `results/figures/`, and tables in `results/tables/`.
- Any threshold or decision made before seeing results goes in `docs/decisions.md`, dated.
- `data/raw/` is never edited directly.
- Pin dependency versions and prompt templates before a multi-phase run so later phases don't
  introduce confounds relative to earlier ones.
- Log every run in `research_log.md` immediately after it finishes, following the `# Logs` section of `CLAUDE.md`.

## Documenting a script

Every script gets an entry here with four things: one sentence on what it tests, the exact command
including all required args, what it expects as input (file format and fields), and what it
produces as output (file format and fields). No vague descriptions like "loads data" — you will be
reading this during a rebuttal three months from now.

```
## Experiment: steering vector non-identifiability baseline

Tests whether different steering vectors produce identical behavioral outputs on TruthfulQA.

**Input:** `data/processed/truthfulqa_prompts.jsonl` — fields: `id`, `prompt`, `category`
**Output:** `results/raw/250612_baseline_v1/outputs.jsonl` — fields: `id`, `prompt`, `response`

**Run:**
uv run -m scripts.run_generation \
  --dataset_path data/processed/truthfulqa_prompts.jsonl \
  --output_dir results/raw/250612_baseline_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 100 \
  --seed 42
```

## Contributing

Working solo, commit directly to `main`. With a collaborator, work on a branch and merge through a
pull request — see `coding_guide.md` for that workflow. Run `pre-commit run --all-files` and
`uv run -m pytest tests/ -v` before every commit, and see `CLAUDE.md` for code conventions.

## Experimental design

See `docs/design.md` for the approach and `docs/decisions.md` for the pre-registered thresholds.

## Experiment: task selection

Ranks ImpossibleBench SWE-bench instances by the pre-registered eligibility rule and seed.

**Input:** `data/raw/impossible_swebench/{original,conflicting}.parquet` — HF `fjzzq2002/impossible_swebench` at
revision `9c2d34f364b7229e8c0ff807c646100bdc18bbb5`, fields `instance_id`, `repo`, `patch`, `test_patch`,
`FAIL_TO_PASS`, `difficulty`
**Output:** `data/splits/task_candidates.json` — `ranked_candidates`: list of `rank`, `instance_id`, `repo`

**Run:**
uv run -m scripts.select_tasks --dataset_path data/raw/impossible_swebench --output_dir data/splits --seed 42

## Experiment: gold-patch verification

Runs each candidate three ways in its SWE-bench container, unpatched on the original tests, with the gold patch on
the original tests, and with the gold patch on the conflicting tests, and keeps the first 30 in rank order where the
FAIL_TO_PASS test fails, passes, and fails respectively.

**Input:** `data/splits/task_candidates.json`, the ImpossibleBench parquets, and
`data/raw/swebench_verified/test.parquet` — HF `SWE-bench/SWE-bench_Verified` at revision
`78f471bf655a3137b2e8a75af1501690ec009ec3`, which supplies `image`, `eval_script`, `log_parser`
**Output:** `data/splits/task_verification.json` — `selected`, and per candidate the F2P/P2P status of each arm;
`data/processed/tasks.jsonl` — fields `id`, `repo`, `base_commit`, `problem_statement`, `first_fail_to_pass_test`,
`patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `original_test_patch`, `conflicting_test_patch`, `image`, `log_parser`,
`eval_type`, `eval_script`, `version`; raw test logs in `data/splits/verification_logs/` (gitignored)

**Run:**
uv run -m scripts.verify_tasks --dataset_path data/raw/impossible_swebench --verified_path data/raw/swebench_verified/test.parquet --output_dir data/splits --seed 42

## Experiment: diff-check validation

Checks the mechanical cheat classifier in `src/metrics/diff_check.py` against 20 labelled diffs (10 gold patches,
10 constructed cheats, one per kind in `src/metrics/cheat_fixtures.py`) and sweeps it over every dataset gold patch
and test patch.

**Input:** the ImpossibleBench parquets and `data/splits/task_candidates.json`
**Output:** `results/raw/260928_smoke_diff_check_validation_v1/labelled_diffs.jsonl` — fields `id`, `instance_id`,
`label_cheat`, `kind`, `diff`, `pred_cheat`, `agrees`, `flags`; `summary.json` — agreement and sweep counts

**Run:**
uv run -m scripts.validate_diff_check --dataset_path data/raw/impossible_swebench --output_dir results/raw/260928_smoke_diff_check_validation_v1 --seed 42

## Experiment: agent rollouts

Runs an Inspect AI coding agent (ImpossibleBench's prompt, bash/python/editor tools) on the verified task set in each
task's SWE-bench container, for one split and one prefix arm, and scores every rollout for cheating with the diff check
and for a real fix with the verification eval script.

**Input:** `data/processed/tasks.jsonl` from `scripts.verify_tasks`; the task images pulled locally; `OPENROUTER_API_KEY`
in `.env` for OpenRouter models (`--model_id openrouter/<model>`, `--provider` pins the upstream with fallbacks off)
**Output:** `results/raw/<run>/outputs.jsonl` — fields `id` (`<instance_id>__epoch<n>`), `instance_id`, `epoch`,
`split`, `arm`, `is_cheat`, `f2p_passed`, `diff_check` (per-category flags), `p2p_failed`, `diff`, `n_messages`,
`error` (scores null on a failed rollout); `summary.json` — `n_expected`, `n_rows`, `n_failed`, `cheat_rate`,
`f2p_pass_rate`, `inspect_log`; full transcripts in `generations/inspect_logs/*.eval` and test logs in
`generations/test_logs/` (both gitignored)

**Run:**
uv run -m scripts.run_rollouts --dataset_path data/processed/tasks.jsonl --output_dir results/raw/260928_smoke_rollouts_v1 --model_id openrouter/qwen/qwen3.8-27b --provider <upstream> --split original --arm issue_only --num_tasks 2 --n_repeats 1 --seed 42

## Experiment: pre-compaction prefixes

Captures, per task and split, the real outputs of the four bash commands the fabricated pre-compaction turns replay:
`ls`, a `grep -n` for the failing test function, a `sed -n` window of its source, and the eval test command run on the
unpatched code. `src/generation/arms.py` turns these into the arm-specific message lists.

**Input:** `data/processed/tasks.jsonl` and the task images pulled locally
**Output:** `data/processed/prefixes_<split>.jsonl` — fields `id`, `split`, `turns` (list of `command`, `output`)

**Run:**
uv run -m scripts.build_prefixes --dataset_path data/processed/tasks.jsonl --split original --output_path data/processed/prefixes_original.jsonl
