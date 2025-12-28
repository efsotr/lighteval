from __future__ import annotations

from typing import Iterable

from datasets import load_dataset

# Template used to format MATH-500 prompts with an explicit ANSWER line requirement.
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
    raise KeyError("Dataset example must contain either 'answer' or 'solution' field, but neither was found.")


def get_prompts(split: str = "test", dataset: Iterable[dict] | None = None) -> list[dict[str, str]]:
    """
    Load the MATH-500 dataset and return a list of formatted prompts with their answers.

    Args:
        split: Dataset split to load (defaults to "test").
        dataset: Optional pre-loaded iterable of examples for testing or custom data.

    Returns:
        inputs: List of {"prompt": PROMPT, "answer": ANSWER}.

    Raises:
        KeyError: If an example is missing both "answer" and "solution" fields.
    """
    data = dataset if dataset is not None else load_dataset("HuggingFaceH4/MATH-500", split=split)

    inputs: list[dict[str, str]] = []
    for example in data:
        prompt = MATH_QUERY_TEMPLATE.format(prompt=example["problem"])
        answer = _extract_answer(example)
        inputs.append({"prompt": prompt, "answer": answer})

    return inputs
