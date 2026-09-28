# scores a generation run's outputs.jsonl and writes summary metrics beside it.
# uv run -m scripts.run_metrics --dataset_path results/raw/250612_example_v1/outputs.jsonl --output_dir results/raw/250612_example_v1 --model_id claude-sonnet-4-6 --seed 42

import logging
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def score(records: list[dict]) -> dict:
    responses = [r["response"] for r in records]
    lengths = [len(response) for response in responses if response is not None]
    return {
        "n_records": len(records),
        "n_failed_responses": sum(1 for response in responses if response is None),
        "mean_response_chars": float(np.mean(lengths)) if lengths else 0.0,
        "median_response_chars": float(np.median(lengths)) if lengths else 0.0,
        "n_empty_responses": sum(1 for length in lengths if length == 0),
    }


def main():
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    records = load_jsonl(config.dataset_path, config.num_tasks)
    LOGGER.info(f"scoring {len(records)} records from {config.dataset_path}")

    metrics = score(records) | {"model_id": config.model_id, "seed": config.seed}
    if metrics["n_failed_responses"]:
        LOGGER.error(f"{metrics['n_failed_responses']} failed responses in {config.dataset_path}")
    if metrics["n_empty_responses"]:
        LOGGER.warning(f"{metrics['n_empty_responses']} empty responses in {config.dataset_path}")

    path = write_json(metrics, output_dir / "metrics.json")
    print(f"Metrics saved to {path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


main()
