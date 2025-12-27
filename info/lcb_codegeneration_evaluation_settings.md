# LCB Code Generation Evaluation Settings

## Task Overview
LiveCodeBench (LCB) collects problems from periodic contests on LeetCode, AtCoder, and Codeforces platforms and uses them for constructing a holistic benchmark for evaluating Code LLMs in code generation scenarios continuously over time.

## Evaluation Configuration

### Basic Configuration
```python
name = "lcb:codegeneration"  # Base configuration (v4_v5)
# Other variants: "lcb:codegeneration_v1", "lcb:codegeneration_v2", etc.
version = 0
```

### Dataset Configuration
```python
hf_repo = "lighteval/code_generation_lite"
hf_subset = "v4_v5"  # Default subset, multiple versions available
hf_avail_splits = ["test"]
evaluation_splits = ["test"]
```

### Available Dataset Versions
```python
configs = [
    "release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6",
    "release_latest",
    "v1", "v2", "v3", "v4", "v5", "v6",
    "v1_v2", "v1_v3", "v1_v4", "v1_v5",
    "v2_v3", "v2_v4", "v2_v5",
    "v3_v4", "v3_v5",
    "v4_v5"  # Default configuration
]
```

### Few-Shot Configuration
```python
few_shots_split = None  # No few-shot split specified
few_shots_select = None  # No few-shot selection method
num_fewshots = 0  # Default value, typically 0 for code generation tasks
```

**Setting Few-Shot Examples:**
- The default `num_fewshots` is **0** (zero-shot evaluation)
- Code generation tasks typically use zero-shot evaluation
- To use few-shot examples if needed, specify: `lcb:codegeneration|N` where N is the number
- Format: `task_name|num_fewshots` (e.g., `lcb:codegeneration|0`)

### Generation Configuration
```python
generation_size = 32768  # Large context for code generation
stop_sequence = []  # no stop sequence, will use EOS token
```

### Prompt Preparation
```python
def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin, solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    
    return query
```

### Prompt Function
```python
def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration") -> Doc:
    query = prepare_prompt(line)
    
    # Parse test cases
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )
```

### Code Extraction and Evaluation
```python
class CodegenMetric(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """Estimates the Pass@1 metric for the code generation task.
        Extract the code from each prediction, runs it for each sample and generation,
        and computes the Pass@1 over the outputs.
        """
        assert doc.specific is not None, "Doc specific field is required for codegen_metric"

        predictions = model_response.final_text
        # Extract generated code snippets
        generated_code_snippets = [[extract_code(pred) for pred in predictions]]
        
        evaluation_sample = {
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        
        evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

        metrics, _ = codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],  # Only run for Pass@1
            num_process_evaluate=8,
        )
        return metrics["pass@1"]
```

### Metrics
```python
lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

metrics = [Metrics.lcb_codegen_metric]
```

## Complete Configuration Object (Base v4_v5)
```python
lcb_codegeneration = LightevalTaskConfig(
    name="lcb:codegeneration",
    prompt_function=lcb_codegeneration_prompt_fn,
    hf_repo="lighteval/code_generation_lite",
    hf_subset="v4_v5",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    generation_size=32768,
    metrics=[Metrics.lcb_codegen_metric],
    stop_sequence=[],
    version=0,
)
```

## Multiple Task Variants
The LCB code generation benchmark provides multiple task variants for different dataset versions:

```python
tasks = []
for subset in configs:
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="lighteval/code_generation_lite",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metrics=[Metrics.lcb_codegen_metric],
        stop_sequence=[],
        version=0,
    )
    tasks.append(task)
```

## Key Features
- **Language**: English
- **Task Type**: Code generation and evaluation
- **Evaluation Split**: Test set only
- **Generation Size**: 32768 tokens (very large for complex code)
- **Scoring**: Pass@1 metric with code execution
- **Test Cases**: Both public and private test cases
- **Evaluation Method**: Execute generated code against test cases
- **Multiprocessing**: Uses 8 processes for evaluation
- **Code Format**: Python code enclosed in markdown code blocks
- **Dataset Versions**: Multiple versions tracking contests over time
- **Platforms**: Problems from LeetCode, AtCoder, and Codeforces
- **Paper**: https://livecodebench.github.io/
