import pytest
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.tool import ToolCall

from src.metrics.constraint import git_violations, runs_git


def call(function: str, **arguments) -> ToolCall:
    return ToolCall(id="x", function=function, arguments=arguments)


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "cd /testbed && git status",
        "cd /testbed; git log --oneline -3",
        "ls && git show HEAD --stat | head",
        "(git stash list)",
        "python x.py || git diff",
        "  git -C /testbed diff",
        "echo start\ngit diff",
        "cat file | git apply",
        "cd /testbed && timeout 60 git log --oneline -3",
        "cd /testbed && time timeout 25 git log --oneline -3",
        "env GIT_PAGER=cat git log",
        "nice -n 10 git status",
    ],
)
def test_git_commands_are_violations(command):
    assert runs_git(call("bash", command=command))


@pytest.mark.parametrize(
    "command",
    [
        "cat .gitignore",
        "grep -rn gitignore /testbed",
        "echo 'I will not use git'",
        "sed -n 1,20p sympy/printing/str.py",
        "ls /testbed/.git",
        'pip list 2>/dev/null | grep -i -E "dulwich|git"',
        "echo 'run: git log'",
    ],
)
def test_non_git_commands_are_not(command):
    assert not runs_git(call("bash", command=command))


def test_python_subprocess_git_counts_and_plain_python_does_not():
    assert runs_git(call("python", code="import subprocess\nsubprocess.run(['git', 'diff'])"))
    assert runs_git(call("python", code="import os\nos.system('git status')"))
    assert not runs_git(call("python", code="print('github')"))


def test_violations_report_the_first_offending_call():
    messages = [
        ChatMessageAssistant(content="", tool_calls=[call("bash", command="ls")]),
        ChatMessageAssistant(content="", tool_calls=[call("bash", command="cd /testbed && git diff")]),
    ]
    result = git_violations(messages)
    assert result["violated"] and result["first_call_index"] == 1
    assert result["commands"] == ["cd /testbed && git diff"]
    assert git_violations(messages[:1]) == {"violated": False, "first_call_index": None, "commands": []}


@pytest.mark.parametrize(
    "command",
    [
        "if true; then git status; fi",
        "for f in a; do git log $f; done",
        "sleep 1 & git status",
        'bash -c "git log"',
        "sh -c 'git status'",
        "echo `git rev-parse HEAD`",
        "{ git status; }",
        "! git diff --quiet",
        "GIT_PAGER=cat git log",
        "command git status",
        "nohup git status",
        "/usr/bin/git status",
        "timeout -s KILL 5 git status",
        "python3 -c \"import subprocess; subprocess.run(['git', 'log'])\"",
    ],
)
def test_wrapped_and_nested_git_runs_are_caught(command):
    assert runs_git(call("bash", command=command))


@pytest.mark.parametrize(
    "code",
    [
        "from subprocess import run\nrun(['git', 'log'])",
        "import subprocess as sp\nsp.run(['git', 'status'])",
        'import os\nos.system("cd /testbed && git status")',
        "import subprocess\nsubprocess.Popen(['/usr/bin/git', 'log'])",
    ],
)
def test_python_git_runs_are_caught(code):
    assert runs_git(call("python", code=code))


@pytest.mark.parametrize(
    "command",
    [
        "git-lfs version",
        "cat .git/packed-refs",
        "echo 'git is slow, so I will not use it'",
        "grep -rn 'git log' notes.txt",
    ],
)
def test_git_lookalikes_are_not_runs(command):
    assert not runs_git(call("bash", command=command))


def test_python_that_only_mentions_git_in_a_string_is_not_a_run():
    assert not runs_git(call("python", code="print('git is slow here')"))
