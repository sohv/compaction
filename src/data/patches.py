import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

TEST_DIRS = {"tests", "test_utils"}
CONFIG_FILES = {"tox.ini", "pytest.ini", "setup.cfg", "pyproject.toml", "conftest.py", ".coveragerc", "Makefile"}
CONFIG_PREFIXES = (".github/", ".circleci/", ".azure-pipelines/", "azure-pipelines", ".travis", ".gitlab-ci")


@dataclass
class FilePatch:
    path: str
    is_new: bool = False
    is_deleted: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1]
    # pytest keeps its own suite under a top-level testing/, while sympy/testing/ is library code
    in_test_dir = any(part in TEST_DIRS for part in parts[:-1]) or parts[0] == "testing"
    return in_test_dir or name.startswith("test_") or name.endswith("_test.py") or name in {"tests.py", "conftest.py"}


def is_config_path(path: str) -> bool:
    return PurePosixPath(path).name in CONFIG_FILES or path.startswith(CONFIG_PREFIXES)


def parse_patch(patch: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    for block in re.split(r"^diff --git ", patch, flags=re.MULTILINE)[1:]:
        header, _, body = block.partition("\n")
        # a/x b/y header; the b-side path is the post-change name, a-side survives deletion
        a_path, b_path = re.match(r"a/(\S+) b/(\S+)", header).groups()
        file_patch = FilePatch(
            path=b_path,
            is_new="\nnew file mode" in "\n" + body.split("@@")[0],
            is_deleted="\ndeleted file mode" in "\n" + body.split("@@")[0],
        )
        in_hunk = False
        for line in body.splitlines():
            if line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+") and not line.startswith("+++"):
                file_patch.added.append(line[1:])
            elif in_hunk and line.startswith("-") and not line.startswith("---"):
                file_patch.removed.append(line[1:])
        if a_path != b_path:
            files.append(FilePatch(path=a_path, is_deleted=True))
        files.append(file_patch)
    return files
