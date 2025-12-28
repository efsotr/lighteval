from __future__ import annotations

from typing import Sequence

import lighteval.tasks.tasks.ifeval.instructions_registry as instructions_registry
from lighteval.tasks.tasks.ifeval.main import _preprocess_response


def _score_sample(sample: dict, response: str) -> dict:
    instruction_list = sample["instruction_id_list"]
    kwargs_list = sample["kwargs"]
    prompt = sample["prompt"]
    all_responses = _preprocess_response(response)

    is_following_list_strict = []
    is_following_list_loose = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        task_kwargs = {k: v for k, v in kwargs_list[index].items() if v}
        instruction.build_description(**task_kwargs)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list_strict.append(True)
        else:
            is_following_list_strict.append(False)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list_loose.append(is_following)

    return {
        "prompt_level_strict_acc": int(all(is_following_list_strict)),
        "inst_level_strict_acc": is_following_list_strict,
        "prompt_level_loose_acc": int(all(is_following_list_loose)),
        "inst_level_loose_acc": is_following_list_loose,
    }


def get_score(inputs: Sequence[dict], outputs: Sequence[str]) -> dict[str, float]:
    """
    Compute IFEval instruction-following metrics for a list of model outputs.

    Args:
        inputs: List of {"prompt": PROMPT, "instruction_id_list": [...], "kwargs": [...]}.
        outputs: List of model outputs corresponding to the prompts.

    Returns:
        Dictionary with aggregated accuracies:
        - IFEval_prompt_level_strict_acc
        - IFEval_prompt_level_loose_acc
        - IFEval_inst_level_strict_acc
        - IFEval_inst_level_loose_acc

    Raises:
        ValueError: If the number of outputs does not match the number of inputs.
    """
    if len(inputs) != len(outputs):
        raise ValueError(
            f"Number of outputs ({len(outputs)}) must match number of inputs ({len(inputs)}) for IFEval scoring."
        )

    if not inputs:
        return {
            "IFEval_prompt_level_strict_acc": 0.0,
            "IFEval_prompt_level_loose_acc": 0.0,
            "IFEval_inst_level_strict_acc": 0.0,
            "IFEval_inst_level_loose_acc": 0.0,
        }

    sample_scores = [_score_sample(sample, response) for sample, response in zip(inputs, outputs)]

    prompt_strict = [score["prompt_level_strict_acc"] for score in sample_scores]
    prompt_loose = [score["prompt_level_loose_acc"] for score in sample_scores]
    inst_strict = [flag for score in sample_scores for flag in score["inst_level_strict_acc"]]
    inst_loose = [flag for score in sample_scores for flag in score["inst_level_loose_acc"]]

    total_instructions = len(inst_strict)
    inst_strict_acc = sum(inst_strict) / total_instructions if total_instructions else 0.0
    inst_loose_acc = sum(inst_loose) / total_instructions if total_instructions else 0.0

    return {
        "IFEval_prompt_level_strict_acc": sum(prompt_strict) / len(prompt_strict),
        "IFEval_prompt_level_loose_acc": sum(prompt_loose) / len(prompt_loose),
        "IFEval_inst_level_strict_acc": inst_strict_acc,
        "IFEval_inst_level_loose_acc": inst_loose_acc,
    }
