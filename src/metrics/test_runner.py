import io
import logging
import shlex
import tarfile
from pathlib import Path

import docker
from swebench.harness.docker_utils import exec_run_with_timeout
from swebench.harness.grading import get_logs_eval, test_failed, test_maintained, test_passed
from swebench.harness.utils import make_test_spec

LOGGER = logging.getLogger(__name__)
PLATFORM = "linux/amd64"


def eval_script_for(verified_row: dict, test_patch: str) -> str:
    """Swap the upstream test patch in the Verified eval script for a variant's test patch."""
    script = verified_row["eval_script"]
    assert script.count(verified_row["test_patch"]) == 1, "upstream test patch must appear exactly once in eval_script"
    return script.replace(verified_row["test_patch"], test_patch)


def put_file(container, text: str, dst: str) -> None:
    data = text.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name=Path(dst).name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    container.put_archive(str(Path(dst).parent), buffer.getvalue())


def ensure_image(client: docker.DockerClient, image: str) -> None:
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        LOGGER.info(f"pulling {image}")
        client.images.pull(image, platform=PLATFORM)


def run_tests(client: docker.DockerClient, image: str, code_patch: str, script: str, timeout: int) -> tuple[str, bool]:
    """Apply code_patch at /testbed, run the eval script, and return the log and whether the patch applied."""
    container = client.containers.run(image, "tail -f /dev/null", detach=True, platform=PLATFORM)
    try:
        if code_patch:
            put_file(container, code_patch, "/tmp/code.diff")
            applied = container.exec_run("git apply -v /tmp/code.diff", workdir="/testbed")
            if applied.exit_code != 0:
                LOGGER.error(
                    f"code patch failed to apply in {image}: {applied.output.decode(errors='replace')[-2000:]}"
                )
                return applied.output.decode(errors="replace"), False
        put_file(container, script, "/eval.sh")
        log, timed_out, seconds = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        LOGGER.info(f"eval in {image} took {seconds:.0f}s, timed_out={timed_out}")
        return log, not timed_out
    finally:
        container.remove(force=True)


def capture_commands(
    client: docker.DockerClient, image: str, test_patch: str, setup: str, commands: list[str], timeout: int
) -> list[str]:
    """Run setup once, then each command in /testbed, returning the combined output the agent's bash tool would show."""
    container = client.containers.run(image, "tail -f /dev/null", detach=True, platform=PLATFORM)
    try:
        put_file(container, test_patch, "/tmp/test_patch.diff")
        result = container.exec_run(["bash", "-c", setup], workdir="/testbed")
        assert result.exit_code == 0, f"setup failed in {image}: {result.output.decode(errors='replace')[-2000:]}"
        outputs = []
        for command in commands:
            # the agent's bash tool runs inside the image's conda env, so the replayed output must too
            script = f"source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed && {command}"
            log, timed_out, _ = exec_run_with_timeout(container, f"bash -c {shlex.quote(script)}", timeout)
            assert not timed_out, f"{command!r} timed out in {image}"
            outputs.append(log)
        return outputs
    finally:
        container.remove(force=True)


def f2p_status(verified_row: dict, script: str, log: str, log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log)
    spec = make_test_spec({**verified_row, "eval_script": script})
    status_map, is_valid = get_logs_eval(spec, str(log_path))
    return {
        "log_valid": is_valid,
        "f2p_passed": [t for t in spec.FAIL_TO_PASS if test_passed(t, status_map)],
        "f2p_failed": [t for t in spec.FAIL_TO_PASS if test_failed(t, status_map)],
        "p2p_failed": [t for t in spec.PASS_TO_PASS if not test_maintained(t, status_map)],
    }
