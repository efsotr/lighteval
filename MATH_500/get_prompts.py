from __future__ import annotations

from typing import Iterable

from datasets import load_dataset

MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()


def _extract_answer(example: dict) -> str:
    if "answer" in example:
        return example["answer"]
    if "solution" in example:
        return example["solution"]
    raise KeyError("Expected either 'answer' or 'solution' in the dataset example.")


def get_prompts(split: str = "test", dataset: Iterable[dict] | None = None) -> list[dict[str, str]]:
    """
    Load the MATH-500 dataset and return a list of formatted prompts with their answers.

    Returns:
        inputs: List of {"prompt": PROMPT, "answer": ANSWER}
    """
    data = dataset if dataset is not None else load_dataset("HuggingFaceH4/MATH-500", split=split)

    inputs: list[dict[str, str]] = []
    for example in data:
        prompt = MATH_QUERY_TEMPLATE.format(prompt=example["problem"])
        answer = _extract_answer(example)
        inputs.append({"prompt": prompt, "answer": answer})

    return inputs
