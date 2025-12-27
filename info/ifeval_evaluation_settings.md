# IFEval Evaluation Settings

## Task Overview
IFEval (Instruction Following Evaluation) is a task that evaluates whether model outputs follow specific format rules and instructions. Unlike typical tasks, it doesn't check for specific outputs but tests if the format obeys predefined rules.

## Evaluation Configuration

### Basic Configuration
```python
name = "ifeval"
version = "0.1"
```

### Dataset Configuration
```python
hf_repo = "google/IFEval"
hf_subset = "default"
hf_avail_splits = ["train"]
evaluation_splits = ["train"]
```

### Few-Shot Configuration
```python
few_shots_split = "train"
few_shots_select = "random_sampling"
num_fewshots = 0  # Default value, can be overridden at runtime
```

**Setting Few-Shot Examples:**
- The default `num_fewshots` is **0** (zero-shot evaluation)
- To use few-shot examples, specify the number when running the task: `ifeval|3` (for 3-shot)
- Few-shot examples are randomly sampled from the train split
- Format: `task_name|num_fewshots` (e.g., `ifeval|0`, `ifeval|3`, `ifeval|5`)

### Generation Configuration
```python
generation_size = 1280
stop_sequence = []  # no stop sequence, will use eot token
```

### Prompt Function
```python
def ifeval_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "instructions_id_list": line["instruction_id_list"],
            "kwargs": line["kwargs"]
        },
    )
```

### Sample Processing
```python
def record_to_sample(record):
    metadata = {
        "instruction_id_list": record["instruction_id_list"],
        "kwargs": record["kwargs"]
    }
    return Sample(
        input=record["prompt"],
        metadata=metadata,
    )
```

### Response Preprocessing
```python
def _preprocess_response(response: str) -> str:
    all_responses = []
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    return all_responses
```

### Solver Configuration
```python
solver = [generate(cache=True)]
scorer = ifeval_scorer()
```

### Metrics
The evaluation uses custom IFEval metrics that compute instruction-following accuracy:

```python
submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc", 
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]

ifeval_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better=dict.fromkeys(submetric_names, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFEvalMetrics(),
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)
```

### IFEval Scorer
```python
@scorer(
    metrics={
        "prompt_level_strict_acc": [accuracy(), stderr()],
        "prompt_level_loose_acc": [accuracy(), stderr()],
    }
)
def ifeval_scorer():
    async def score(state: TaskState, target: Target):
        response = state.output.completion
        instruction_list = state.metadata["instruction_id_list"]
        all_kwargs = state.metadata["kwargs"]
        prompt = state.input
        all_responses = _preprocess_response(response)

        is_following_list_strict = []
        is_following_list_loose = []
        
        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
            instruction.build_description(**task_kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
            
            # Strict checking
            if response.strip() and instruction.check_following(response):
                is_following_list_strict.append(True)
            else:
                is_following_list_strict.append(False)
            
            # Loose checking
            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break
            is_following_list_loose.append(is_following)
        
        return Score(
            value={
                "prompt_level_strict_acc": int(all(is_following_list_strict)),
                "prompt_level_loose_acc": int(all(is_following_list_loose)),
            },
            explanation=str(instruction_list),
        )
    return score
```

## Complete Configuration Object
```python
ifeval = LightevalTaskConfig(
    name="ifeval",
    prompt_function=ifeval_prompt,
    hf_repo="google/IFEval",
    hf_subset="default",
    metrics=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],
    version="0.1",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=ifeval_scorer(),
)
```

## Key Features
- **Language**: English
- **Task Type**: Instruction-following evaluation
- **Evaluation Split**: Train set (used as test set)
- **Generation Size**: 1280 tokens
- **Scoring**: Custom instruction-following metrics (strict and loose)
- **Metric Types**: 
  - Prompt-level accuracy (all instructions must be followed)
  - Instruction-level accuracy (individual instruction compliance)
  - Strict mode (exact compliance)
  - Loose mode (allows some variations in formatting)
- **Dependencies**: Requires `langdetect` package
- **Paper**: https://arxiv.org/abs/2311.07911
