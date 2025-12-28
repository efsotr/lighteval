from __future__ import annotations

from typing import Iterable

from datasets import load_dataset


def get_prompts(split: str = "train", dataset: Iterable[dict] | None = None) -> list[dict]:
    """
    Load the IFEval dataset and return a list of formatted prompts with their instruction metadata.

    Args:
        split: Dataset split to load (defaults to "train").
        dataset: Optional pre-loaded iterable of examples for testing or custom data.

    Returns:
        inputs: List of {"prompt": PROMPT, "instruction_id_list": [...], "kwargs": [...]}.
    """
    data = dataset if dataset is not None else load_dataset("google/IFEval", split=split)

    return [
        {
            "prompt": example["prompt"],
            "instruction_id_list": example["instruction_id_list"],
            "kwargs": example["kwargs"],
        }
        for example in data
    ]
