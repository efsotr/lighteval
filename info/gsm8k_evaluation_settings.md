# GSM8K Evaluation Settings

## Task Overview
GSM8K (Grade School Math 8K) is a dataset of 8,000+ high-quality, single-step arithmetic word problems designed to evaluate mathematical reasoning capabilities.

## Evaluation Configuration

### Basic Configuration
```python
name = "gsm8k"
version = 0
```

### Dataset Configuration
```python
hf_repo = "openai/gsm8k"
hf_subset = "main"
hf_avail_splits = ["train", "test"]
evaluation_splits = ["test"]
```

### Few-Shot Configuration
```python
few_shots_split = None
few_shots_select = "random_sampling_from_train"
num_fewshots = 0  # Default value, can be overridden at runtime
```

**Setting Few-Shot Examples:**
- The default `num_fewshots` is **0** (zero-shot evaluation)
- To use few-shot examples, specify the number when running the task: `gsm8k|5` (for 5-shot)
- Few-shot examples are sampled from the train split using `random_sampling_from_train`
- Format: `task_name|num_fewshots` (e.g., `gsm8k|0`, `gsm8k|5`, `gsm8k|10`)

### Generation Configuration
```python
generation_size = 256
stop_sequence = ["Question:"]
```

### Prompt Template
```python
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
```

### Prompt Function
```python
def gsm8k_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {line['answer']}"],
        gold_index=0,
    )
```

### Sample Processing
```python
def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})

def sample_to_fewshot(sample):
    return f"{sample.input}\n\nReasoning:\n" + \
           f"{sample.metadata['reasoning']}\n\n" + \
           f"ANSWER: {sample.target}"
```

### Solver Configuration
```python
solver = [
    prompt_template(MATH_PROMPT_TEMPLATE),
    generate(cache=True)
]
```

### Metrics
```python
metrics = [Metrics.expr_gold_metric]
scorer = math_scorer()
```

## Complete Configuration Object
```python
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=["Question:"],
    version=0,
)
```

## Key Features
- **Language**: English
- **Task Type**: Mathematical reasoning
- **Evaluation Split**: Test set only
- **Generation Size**: 256 tokens
- **Scoring**: Mathematical expression evaluation using `math_scorer()`
- **Paper**: https://arxiv.org/abs/2110.14168
