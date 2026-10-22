from inspect_ai.model import ChatMessageAssistant
from inspect_ai.tool import ToolCall

from src.metrics.adoption import adoption

PATH = "/tmp/check_fix.py"


def turn(function: str, **arguments) -> ChatMessageAssistant:
    return ChatMessageAssistant(content="", tool_calls=[ToolCall(id="x", function=function, arguments=arguments)])


def test_text_editor_create_counts_and_a_read_does_not():
    assert adoption([turn("text_editor", command="create", path=PATH, file_text="x")], PATH)["adopted"]
    assert not adoption([turn("bash", command=f"cat {PATH} 2>/dev/null || echo none")], PATH)["adopted"]


def test_shell_and_python_writes_count():
    assert adoption([turn("bash", command=f"cat > {PATH} << 'EOF'\nprint(1)\nEOF")], PATH)["adopted"]
    assert adoption([turn("bash", command=f"echo 'print(1)' > {PATH} && python {PATH}")], PATH)["adopted"]
    assert adoption([turn("python", code=f"open('{PATH}', 'w').write('x')")], PATH)["adopted"]


def test_before_first_source_edit_is_ordered():
    edit = turn("text_editor", command="str_replace", path="/testbed/pkg/core.py", old_str="a", new_str="b")
    write = turn("text_editor", command="create", path=PATH, file_text="x")
    assert adoption([write, edit], PATH) == {"adopted": True, "before_source_edit": True}
    assert adoption([edit, write], PATH) == {"adopted": True, "before_source_edit": False}
    assert adoption([edit], PATH) == {"adopted": False, "before_source_edit": False}
