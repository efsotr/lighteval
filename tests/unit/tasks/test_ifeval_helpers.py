from IFEval import get_prompts, metric


def test_get_prompts_formats_entries(monkeypatch):
    dataset = [
        {
            "prompt": "Include foo in the reply.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["foo"]}],
        }
    ]
    monkeypatch.setattr(get_prompts, "load_dataset", lambda *_, **__: dataset)

    inputs = get_prompts.get_prompts()

    assert inputs == dataset


def test_get_score_computes_ifeval_metrics():
    inputs = [
        {
            "prompt": "Include foo in the reply.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["foo"]}],
        }
    ]
    outputs = ["This sentence contains foo twice. And another foo."]

    scores = metric.get_score(inputs, outputs)

    assert scores == {
        "IFEval_prompt_level_strict_acc": 1.0,
        "IFEval_prompt_level_loose_acc": 1.0,
        "IFEval_inst_level_strict_acc": 1.0,
        "IFEval_inst_level_loose_acc": 1.0,
    }
