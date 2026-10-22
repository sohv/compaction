# writes S(T) with the agent model and codex's compaction prompt, resampling replies that are not summaries.
import logging

from inspect_ai.model import Model, ModelOutput
from inspect_ai.tool import bash, python, text_editor, think

from src.generation.continuations import summary_request
from src.generation.swe_agent import THINK_DESCRIPTION

LOGGER = logging.getLogger(__name__)
MIN_SUMMARY_CHARS = 500
MAX_SUMMARY_ATTEMPTS = 3
AGENT_TOOLS = [bash(), python(), text_editor(), think(THINK_DESCRIPTION)]


async def summarize_one(model: Model, messages: list, cut: int) -> tuple[ModelOutput, int]:
    """K3 sometimes answers the compaction prompt by announcing more work instead of summarizing; those are resampled."""
    for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
        output = await model.generate(summary_request(messages, cut), tools=AGENT_TOOLS, tool_choice="none")
        if len(output.completion) >= MIN_SUMMARY_CHARS:
            return output, attempt
        LOGGER.warning(
            f"attempt {attempt}: {len(output.completion)}-char completion is not a summary "
            f"({output.stop_reason=}, {output.usage.output_tokens if output.usage else None} tokens), resampling"
        )
    raise AssertionError(f"no summary of at least {MIN_SUMMARY_CHARS} chars after {MAX_SUMMARY_ATTEMPTS} attempts")
