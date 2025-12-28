from MATH_500 import get_prompts
from MATH_500 import metric


def test_get_prompts_formats_problem(monkeypatch):
    dataset = [{"problem": "What is 1+1?", "answer": "2"}]
    monkeypatch.setattr(get_prompts, "load_dataset", lambda *_, **__: dataset)

    inputs = get_prompts.get_prompts()

    assert inputs == [
        {"prompt": get_prompts.MATH_QUERY_TEMPLATE.format(prompt="What is 1+1?"), "answer": "2"}
    ]


def test_get_score_computes_pass_at_one():
    inputs = [
        {"prompt": "p1", "answer": "2"},
        {"prompt": "p2", "answer": "5"},
    ]
    outputs = ["Reasoning\nANSWER: 2", "Final\nANSWER: 4"]

    scores = metric.get_score(inputs, outputs)

    assert scores == {"MATH_500_pass@1": 0.5}
