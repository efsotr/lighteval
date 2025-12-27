# LCB 任务中 pass@1:16 的分析

## 问题概述

对于 LCB (Live Code Bench) 代码生成任务，指标名称为 `codegen_pass@1:16`。本文档解释了这个 `16` 的含义以及 LCB 是如何设置 `sampling_params` 中的 `n` 参数为 16 的。

## pass@1:16 中 "16" 的含义

### 1. 基本概念

`pass@1:16` 表示 **Pass@k 指标**，其中：
- `k=1`：表示只要有 1 个样本通过测试就算成功
- `n=16`：表示为每个问题生成 16 个不同的代码样本

这是代码生成评估中的标准指标，用于衡量模型生成正确代码的能力。

### 2. Pass@k 的计算方法

Pass@k 使用无偏估计器计算（来自论文 https://arxiv.org/pdf/2107.03374）：

```
pass@k = 1 - (n-c choose k) / (n choose k)
```

其中：
- `n`：总样本数（在这里是 16）
- `c`：通过测试的样本数
- `k`：阈值（在这里是 1）

### 3. 为什么需要生成多个样本？

生成多个样本（n=16）而不是单个样本的原因：
- **提高准确性**：通过多次采样可以更准确地估计模型的性能
- **减少方差**：单个样本的结果可能受随机性影响，多样本可以提供更稳定的评估
- **标准化指标**：pass@k 是代码生成任务的标准评估方法，便于与其他研究比较

## LCB 如何设置 sampling_params 中的 n 为 16

### 1. 代码位置

相关代码位于：
- **主文件**：`src/lighteval/tasks/tasks/lcb/main.py`
- **指标文件**：`src/lighteval/metrics/metrics_sample.py`
- **任务配置**：`src/lighteval/tasks/lighteval_task.py`

### 2. 实现机制

#### 步骤 1：指标定义（main.py 第 107-114 行）

```python
lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",  # 这里的 :16 表示需要 16 个样本
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)
```

#### 步骤 2：提取样本数

框架会解析指标名称 `codegen_pass@1:16`：
- 识别到 `:` 后面的数字 `16` 作为样本数参数
- 这个参数会被传递给采样函数

#### 步骤 3：设置 num_samples

在 `LightevalTask` 类中（lighteval_task.py 第 248-253 行）：

```python
# 初始化时，从每个指标中提取 num_samples
self.num_samples = [1]
for metric in self.cfg.metrics:
    if metric.category == SamplingMethod.GENERATIVE:
        if hasattr(metric.sample_level_fn, 'num_samples'):
            self.num_samples.append(metric.sample_level_fn.num_samples())
```

#### 步骤 4：传递给模型

在生成请求时（data.py 第 246 行）：

```python
# 从 doc.num_samples 获取样本数
num_samples = doc.num_samples
```

#### 步骤 5：应用到 sampling_params

对于不同的模型后端：

**VLLM 模型**（vllm_model.py 第 421 行）：
```python
sampling_params.n = num_samples  # 设置为 16
```

**Transformers 模型**（transformers_model.py 第 805 行）：
```python
num_return_sequences=num_samples  # 设置为 16
```

**SGLang 模型**（sglang_model.py 第 331 行）：
```python
self.sampling_params["n"] = num_samples  # 设置为 16
```

## 当前实现状态

✅ **已修复**：`CodegenMetric` 类现在正确继承了 `SamplingMetric` 并实现了 `num_samples()` 方法。

修复后的实现（main.py 第 79-119 行）：

```python
class CodegenMetric(SamplingMetric, SampleLevelComputation):
    def __init__(self, n: int = 16, **kwargs):
        """初始化代码生成指标
        
        Args:
            n (int): 生成的样本数，默认为 16
            **kwargs: 其他参数传递给 SamplingMetric
        """
        super().__init__(**kwargs)
        self.n = n
        self.attribute_must_be_set = ["n"]
    
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """计算 pass@1 指标"""
        # ... 实现代码生成评估
        return metrics["pass@1"]
    
    def num_samples(self):
        """返回需要生成的样本数"""
        return self.n
```

指标定义（main.py 第 122-129 行）：

```python
lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetric(n=16),  # ✅ 正确设置 n=16
    corpus_level_fn=np.mean,
    batched_compute=False,
)
```

现在框架会正确识别 `CodegenMetric` 是 `SamplingMetric` 的实例，并调用 `num_samples()` 返回 16，从而为每个编程问题生成 16 个代码样本。

### 4. 指标名称约定

指标名称 `codegen_pass@1:16` 遵循以下约定：
- **格式**：`<metric_type>_pass@<k>:<n>`
- **示例**：
  - `pass@1:16` - k=1, n=16
  - `pass@5:100` - k=5, n=100
  - `maj@3:10` - 多数投票@3，从10个样本中选择

这个命名约定虽然清晰，但需要配合正确的 `num_samples()` 实现才能生效。

## 工作流程总结

完整的数据流：

```
1. 任务配置 (main.py)
   ↓
   指标名称: "codegen_pass@1:16"
   
2. 指标创建 (main.py)
   ↓
   CodegenMetric(n=16).num_samples() → 返回 16
   
3. 任务初始化 (lighteval_task.py)
   ↓
   task.num_samples = [1, 16]  # 合并所有指标的样本数
   
4. 文档准备 (lighteval_task.py)
   ↓
   doc.num_samples = max([1, 16]) = 16
   
5. 请求生成 (data.py)
   ↓
   从 doc 中提取 num_samples = 16
   
6. 模型推理 (model.py)
   ↓
   sampling_params.n = 16
   或
   num_return_sequences = 16
   
7. 生成结果
   ↓
   为每个问题生成 16 个不同的代码样本
   
8. 评估 (codegen_metrics.py)
   ↓
   运行代码并测试，计算有多少个通过
   
9. 计算指标 (codegen_metrics.py)
   ↓
   使用 pass@k 估计器计算最终得分
```

## 参考文献

- **Pass@k 论文**：Evaluating Large Language Models Trained on Code (https://arxiv.org/pdf/2107.03374)
- **LiveCodeBench**：https://livecodebench.github.io/
- **LightEval 框架**：https://github.com/huggingface/lighteval

## 总结

- **16 的含义**：表示为每个编程问题生成 16 个不同的代码样本
- **设置方式**：通过指标的 `num_samples()` 方法返回 16，这个值会被传递到模型的 sampling_params 中
- **当前问题**：`CodegenMetric` 类缺少 `num_samples()` 方法的实现
- **解决方案**：需要在 `CodegenMetric` 类中添加 `num_samples()` 方法返回 16
