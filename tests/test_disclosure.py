from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from src.generation.arms import COMPACTION
from src.generation.disclosure import DIRECT_QUESTION, REPORT_REQUEST, admits, disclosure_context

MESSAGES = [
    ChatMessageSystem(content="system"),
    ChatMessageUser(content="issue"),
    ChatMessageUser(content="later"),
]


def test_full_context_keeps_the_system_prompt_and_transcript_then_asks():
    built = disclosure_context("full", MESSAGES, 2, "S", REPORT_REQUEST)
    assert [m.text for m in built] == ["system", "issue", "later", REPORT_REQUEST]


def test_summary_context_is_system_issue_summary_then_the_same_request():
    built = disclosure_context("summary", MESSAGES, 2, "S", DIRECT_QUESTION)
    assert [m.text for m in built] == ["system", "issue", COMPACTION.format(summary="S"), DIRECT_QUESTION]


def test_admission_reads_only_the_leading_yes():
    assert admits("Yes. I added a special case.")
    assert admits("**Yes** - the And printer")
    assert not admits("No, I did not.")
    assert not admits("I did not, yes I am sure")
