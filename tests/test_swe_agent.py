import json
from pathlib import Path

import pytest

from src.generation.swe_agent import (
    build_sample,
    compose_file,
    issue_only_messages,
    system_prompt,
    eval_test_command,
    verified_row,
)

TASKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "tasks.jsonl"


@pytest.fixture
def task() -> dict:
    if not TASKS_PATH.exists():
        pytest.skip("task set not built yet, run scripts.verify_tasks")
    return json.loads(TASKS_PATH.read_text().splitlines()[0])


def test_test_command_is_the_line_between_markers():
    script = "setup\n: '>>>>> Start Test Output'\npytest -rA tests/test_x.py\n: '>>>>> End Test Output'\nteardown"
    assert eval_test_command(script) == "pytest -rA tests/test_x.py"


def test_compose_file_pins_local_image_offline(tmp_path):
    text = Path(compose_file("swebench/sweb.eval.x86_64.a_1776_b-1:latest", tmp_path)).read_text()
    assert "image: swebench/sweb.eval.x86_64.a_1776_b-1:latest" in text
    assert "x-local: true" in text
    assert "network_mode: none" in text


def test_sample_carries_the_split_test_patch(task, tmp_path):
    for split in ("original", "conflicting"):
        sample = build_sample(task, split, issue_only_messages(task), tmp_path)
        assert sample.metadata["test_patch"] == task[f"{split}_test_patch"]
        assert sample.id == task["id"]


def test_system_prompt_names_the_eval_command(task):
    prompt = system_prompt(task)
    assert eval_test_command(task["eval_script"]) in prompt
    assert "{" not in prompt


def test_verified_row_uses_the_upstream_test_patch(task):
    row = verified_row(task)
    assert row["instance_id"] == task["id"]
    assert row["eval_script"].count(row["test_patch"]) == 1
