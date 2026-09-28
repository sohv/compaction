from pathlib import Path

import numpy as np
import pandas as pd

from src.data.patches import is_test_path, parse_patch

DATASET_REVISION = "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"
EASY_DIFFICULTIES = {"<15 min fix", "15 min - 1 hour"}


def load_split(raw_dir: Path | str, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(raw_dir) / f"{split}.parquet").set_index("instance_id", drop=False)


def eligibility(original: pd.DataFrame, conflicting: pd.DataFrame) -> pd.DataFrame:
    gold_files = original.patch.map(lambda p: [f.path for f in parse_patch(p)])
    return pd.DataFrame(
        {
            "instance_id": original.instance_id,
            "repo": original.repo,
            "has_one_f2p": original.FAIL_TO_PASS.map(len) == 1,
            "is_easy": original.difficulty.isin(EASY_DIFFICULTIES),
            "gold_touches_one_source_file": gold_files.map(lambda fs: len(fs) == 1 and not is_test_path(fs[0])),
            "conflicting_differs": conflicting.loc[original.index, "test_patch"] != original.test_patch,
        }
    )


def rank_candidates(eligible: pd.DataFrame, seed: int, max_per_repo: int, n_candidates: int) -> list[str]:
    shuffled = eligible.sample(frac=1, random_state=np.random.RandomState(seed))
    capped = shuffled[shuffled.groupby("repo").cumcount() < max_per_repo]
    return capped.instance_id.head(n_candidates).tolist()
