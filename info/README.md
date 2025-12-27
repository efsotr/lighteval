# Evaluation Settings Documentation

This directory contains detailed evaluation settings and configurations for various tasks in LightEval.

## Available Task Configurations

### 1. GSM8K (Grade School Math 8K)
**File:** [gsm8k_evaluation_settings.md](./gsm8k_evaluation_settings.md)

- **Task Type:** Mathematical reasoning
- **Dataset:** openai/gsm8k
- **Evaluation Split:** test
- **Generation Size:** 256 tokens
- **Metric:** Mathematical expression evaluation (expr_gold_metric)
- **Few-shot:** Random sampling from train set
- **Paper:** https://arxiv.org/abs/2110.14168

**Key Features:**
- Single-step arithmetic word problems
- Uses step-by-step reasoning prompts
- Answer extraction using specific format: "ANSWER: $ANSWER"
- Stop sequence: "Question:"

---

### 2. IFEval (Instruction Following Evaluation)
**File:** [ifeval_evaluation_settings.md](./ifeval_evaluation_settings.md)

- **Task Type:** Instruction-following evaluation
- **Dataset:** google/IFEval
- **Evaluation Split:** train (used as test set)
- **Generation Size:** 1280 tokens
- **Metrics:** 
  - Prompt-level strict accuracy
  - Instruction-level strict accuracy
  - Prompt-level loose accuracy
  - Instruction-level loose accuracy
- **Paper:** https://arxiv.org/abs/2311.07911

**Key Features:**
- Evaluates format and rule compliance
- Dual evaluation mode (strict and loose)
- Response preprocessing with multiple variations
- Custom instruction registry for rule checking
- Requires `langdetect` package

---

### 3. LCB Code Generation (LiveCodeBench)
**File:** [lcb_codegeneration_evaluation_settings.md](./lcb_codegeneration_evaluation_settings.md)

- **Task Type:** Code generation and evaluation
- **Dataset:** lighteval/code_generation_lite
- **Evaluation Split:** test
- **Generation Size:** 32768 tokens (very large context)
- **Metric:** Pass@1 (code execution success rate)
- **Paper:** https://livecodebench.github.io/

**Key Features:**
- Problems from LeetCode, AtCoder, and Codeforces
- Multiple dataset versions (v1-v6, combined versions)
- Default version: v4_v5
- Public and private test cases
- Code execution with 8 parallel processes
- Python code generation in markdown format
- Continuous benchmark updates over time

---

## Quick Comparison Table

| Task | Type | Dataset | Gen Size | Primary Metric | Key Challenge |
|------|------|---------|----------|----------------|---------------|
| gsm8k | Math | openai/gsm8k | 256 | expr_gold | Multi-step reasoning |
| ifeval | Instruction | google/IFEval | 1280 | Accuracy (strict/loose) | Format compliance |
| lcb:codegeneration | Code | lighteval/code_generation_lite | 32768 | Pass@1 | Code correctness |

## Usage

To use these tasks in LightEval, reference them by their task names:
- `gsm8k`
- `ifeval`
- `lcb:codegeneration` (or specific versions like `lcb:codegeneration_v5`)

For detailed configuration parameters, solver pipelines, and metric implementations, refer to the individual configuration files.

## File Locations

The actual implementation files can be found at:
- GSM8K: `/src/lighteval/tasks/tasks/gsm8k.py`
- IFEval: `/src/lighteval/tasks/tasks/ifeval/main.py`
- LCB: `/src/lighteval/tasks/tasks/lcb/main.py`
