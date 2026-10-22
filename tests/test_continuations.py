from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

from src.generation.arms import COMPACTION, USER_LINE
from src.generation.continuations import (
    BENIGN_LINES,
    BENIGN_PATH,
    CODEX_COMPACT_PROMPT,
    SKIP_LINE,
    SUMMARY_CONSTRAINT_LINE,
    TOTAL_MESSAGES,
    continuation_messages,
    find_cut,
    first_pass_cut,
    message_budget,
    replay_calls,
    summary_request,
)

TASK = {"FAIL_TO_PASS": ["pkg/tests/test_core.py::test_parse"], "problem_statement": "parse drops entries"}


def call(i: int, function: str, **arguments) -> ToolCall:
    return ToolCall(id=f"c{i}", function=function, arguments=arguments)


MESSAGES = [
    ChatMessageSystem(content="system"),
    ChatMessageUser(content="Please solve the following coding issue:\n\nparse drops entries"),
    ChatMessageAssistant(content="look", tool_calls=[call(0, "bash", cmd="grep -n test_parse pkg/tests/test_core.py")]),
    ChatMessageTool(content="3:def test_parse():", tool_call_id="c0", function="bash"),
    ChatMessageAssistant(
        content="",
        tool_calls=[
            call(1, "think", thought="hmm"),
            call(2, "text_editor", command="create", path="/tmp/r.py", file_text="x"),
        ],
    ),
    ChatMessageTool(content="", tool_call_id="c1", function="think"),
    ChatMessageTool(content="created", tool_call_id="c2", function="text_editor"),
    ChatMessageAssistant(content="run", tool_calls=[call(3, "bash", cmd="pytest pkg/tests/test_core.py")]),
    ChatMessageTool(
        content="FAILED pkg/tests/test_core.py::test_parse - assert [] == ['a']", tool_call_id="c3", function="bash"
    ),
    ChatMessageAssistant(content="the test is wrong"),
]


def test_cut_is_the_first_failing_run_of_the_target_test_not_a_grep_hit():
    assert find_cut(MESSAGES, TASK) == 8


def test_replay_covers_every_state_changing_call_before_the_cut_but_not_the_cut_call_or_think():
    calls = replay_calls(MESSAGES, 8)
    assert [c.id for c in calls] == ["c0", "c2"]


def test_budget_is_what_the_original_rollout_had_left():
    assert message_budget(8) == TOTAL_MESSAGES - 9


def test_summary_request_appends_the_codex_prompt_verbatim_after_the_cut():
    request = summary_request(MESSAGES, 8)
    assert request[:9] == MESSAGES[:9]
    assert request[-1].role == "user" and request[-1].text == CODEX_COMPACT_PROMPT
    assert CODEX_COMPACT_PROMPT.startswith("You are performing a CONTEXT CHECKPOINT COMPACTION.")


def test_full_arm_is_the_transcript_to_the_cut_without_the_system_prompt():
    assert continuation_messages("full", MESSAGES, 8, "S") == MESSAGES[1:9]


def test_summary_arms_differ_only_in_the_planted_line():
    issue = MESSAGES[1].text
    summary = continuation_messages("summary", MESSAGES, 8, "S")
    skip = continuation_messages("summary_skip", MESSAGES, 8, "S")
    user_skip = continuation_messages("summary_user_skip", MESSAGES, 8, "S")
    assert [m.text for m in summary] == [issue, COMPACTION.format(summary="S")]
    assert [m.text for m in skip] == [issue, COMPACTION.format(summary=f"S\n\n{SKIP_LINE}")]
    assert [m.text for m in user_skip] == [issue, COMPACTION.format(summary="S"), USER_LINE]


def test_a_file_view_that_shows_the_test_and_an_assert_is_not_a_cut():
    view = ChatMessageTool(
        content="def test_parse():\n    assert parse('a') == ['a']", tool_call_id="c0", function="bash"
    )
    assert find_cut([*MESSAGES[:3], view, *MESSAGES[4:]], TASK) == 8


def test_budgeted_limit_counts_from_after_the_system_prompt():
    # full arm: system + transcript to the cut is cut + 1 messages, so the cap lands on the original 150
    assert (8 + 1) + message_budget(8) == TOTAL_MESSAGES


def test_a_grep_whose_output_names_the_test_next_to_an_error_is_not_a_cut():
    grep_call = ChatMessageAssistant(content="", tool_calls=[call(0, "bash", command="grep -n test_parse -A3 f.py")])
    grep_out = ChatMessageTool(
        content="3:def test_parse():\n4:    raises(ValueError)", tool_call_id="c0", function="bash"
    )
    assert find_cut([*MESSAGES[:2], grep_call, grep_out, *MESSAGES[4:]], TASK) == 8


def test_benign_arms_deliver_the_same_decision_from_summary_user_and_own_think_call():
    issue = MESSAGES[1].text
    summary = continuation_messages("summary_benign", MESSAGES, 8, "S")
    user = continuation_messages("user_benign", MESSAGES, 8, "S")
    own = continuation_messages("self_benign", MESSAGES, 8, "S")
    assert [m.text for m in summary] == [issue, COMPACTION.format(summary=f"S\n\n{BENIGN_LINES['summary']}")]
    assert [m.text for m in user] == [issue, COMPACTION.format(summary="S"), BENIGN_LINES["user"]]
    assert [m.role for m in own] == ["user", "user", "assistant", "tool"]
    assert own[2].tool_calls[0].function == "think"
    assert own[2].tool_calls[0].arguments == {"thought": BENIGN_LINES["self"]}
    assert all(BENIGN_PATH in line for line in BENIGN_LINES.values())


def test_summary_constraint_appends_the_constraint_as_the_last_summary_line():
    issue = MESSAGES[1].text
    built = continuation_messages("summary_constraint", MESSAGES, 8, "S")
    assert [m.text for m in built] == [issue, COMPACTION.format(summary=f"S\n\n{SUMMARY_CONSTRAINT_LINE}")]


def test_pass_cut_is_the_first_test_run_where_the_target_test_passes():
    passing = [
        ChatMessageAssistant(content="", tool_calls=[call(9, "bash", command="pytest -rA pkg/tests/test_core.py")]),
        ChatMessageTool(
            content="PASSED pkg/tests/test_core.py::test_parse\n1 passed", tool_call_id="c9", function="bash"
        ),
    ]
    assert first_pass_cut(MESSAGES, TASK) is None
    assert first_pass_cut([*MESSAGES, *passing], TASK) == len(MESSAGES) + 1


def test_a_run_where_the_target_line_fails_is_not_a_pass_even_if_others_pass():
    mixed = [
        ChatMessageAssistant(content="", tool_calls=[call(9, "bash", command="bin/test pkg/tests/test_core.py")]),
        ChatMessageTool(content="test_other ok\ntest_parse F\n[FAIL]", tool_call_id="c9", function="bash"),
    ]
    assert first_pass_cut([*MESSAGES[:2], *mixed], TASK) is None
    sympy_style = [
        mixed[0],
        ChatMessageTool(content="test_other ok\ntest_parse ok\n[OK]", tool_call_id="c9", function="bash"),
    ]
    assert first_pass_cut([*MESSAGES[:2], *sympy_style], TASK) == 3


def test_pass_cut_matches_the_test_name_as_a_whole_word_only():
    task = {"FAIL_TO_PASS": ["test_Printer"], "problem_statement": "x"}
    run = ChatMessageAssistant(
        content="", tool_calls=[call(9, "bash", command="bin/test sympy/printing/tests/test_x.py")]
    )
    prefix_only = ChatMessageTool(
        content="test_Printer_standard ok\ntest_Printer F", tool_call_id="c9", function="bash"
    )
    assert first_pass_cut([*MESSAGES[:2], run, prefix_only], task) is None
    real = ChatMessageTool(content="test_Printer ok\ntest_Printer_standard ok", tool_call_id="c9", function="bash")
    assert first_pass_cut([*MESSAGES[:2], run, real], task) == 3
