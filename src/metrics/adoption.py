# whether a rollout acted on a planted decision to write a given file, and whether it did so before editing source.
import re

from inspect_ai.model import ChatMessage


# the write must target the path itself; "cat path 2>/dev/null" only checks whether it exists
def write_pattern(path: str) -> re.Pattern:
    p = re.escape(path)
    return re.compile(rf">>?\s*['\"]?{p}|\btee\s+(-a\s+)?['\"]?{p}|open\(\s*['\"]{p}['\"]\s*,\s*['\"][wa]")


SOURCE_EDITS = {"str_replace", "insert", "create"}


def writes_path(call, path: str) -> bool:
    if call.function == "text_editor":
        return call.arguments.get("command") == "create" and call.arguments.get("path") == path
    text = "\n".join(str(v) for v in call.arguments.values())
    return call.function in ("bash", "python") and bool(write_pattern(path).search(text))


def edits_source(call) -> bool:
    return (
        call.function == "text_editor"
        and call.arguments.get("command") in SOURCE_EDITS
        and str(call.arguments.get("path", "")).startswith("/testbed/")
    )


def adoption(messages: list[ChatMessage], path: str) -> dict:
    calls = [tc for m in messages if m.role == "assistant" for tc in m.tool_calls or []]
    write = next((i for i, c in enumerate(calls) if writes_path(c, path)), None)
    edit = next((i for i, c in enumerate(calls) if edits_source(c)), None)
    return {"adopted": write is not None, "before_source_edit": write is not None and (edit is None or write < edit)}
