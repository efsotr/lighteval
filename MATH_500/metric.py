from __future__ import annotations

from typing import Sequence

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes_inspect,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.utils.language import Language

_GOLD_REGEXES = get_extraction_regexes_inspect(
    (ExprExtractionConfig(), LatexExtractionConfig()), language=Language.ENGLISH
)
_PRED_REGEXES = get_extraction_regexes_inspect(
    (ExprExtractionConfig(), LatexExtractionConfig()), language=Language.ENGLISH
)
_FALLBACK_MODE = "first_match"
_EXTRACTION_MODE = "any_match"
_PRECISION = 6
_TIMEOUT_SECONDS = 5


def _extract_values(text: str, regexes):
    return extract_target_from_pred(text, regexes, _FALLBACK_MODE, _EXTRACTION_MODE, _TIMEOUT_SECONDS)


def _score_prediction(gold_answer: str, prediction: str) -> float:
    gold_extractions = _extract_values(gold_answer, _GOLD_REGEXES) or [gold_answer]
    pred_extractions = _extract_values(prediction, _PRED_REGEXES)

    if not pred_extractions:
        return 0.0

    return (
        1.0
        if compare_gold_target(
            gold_extractions, pred_extractions, precision=_PRECISION, timeout_seconds=_TIMEOUT_SECONDS
        )
        else 0.0
    )


def get_score(inputs: Sequence[dict[str, str]], outputs: Sequence[str]) -> dict[str, float]:
    """
    Compute pass@1 for MATH_500 outputs.

    Args:
        inputs: List of {"prompt": PROMPT, "answer": ANSWER}
        outputs: List of model outputs corresponding to the prompts.
    """
    if len(inputs) != len(outputs):
        raise ValueError("Number of outputs must match number of inputs for MATH_500 scoring.")

    if not inputs:
        return {"MATH_500_pass@1": 0.0}

    correct = sum(_score_prediction(str(sample["answer"]), str(output)) for sample, output in zip(inputs, outputs))
    return {"MATH_500_pass@1": correct / len(inputs)}
