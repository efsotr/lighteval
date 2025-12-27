# LightEval 数据集加载机制分析 - 以 GSM8K 为例

## 调查背景

本文档分析 lighteval 如何加载数据集，以 gsm8k 为例，并探究为什么使用 HuggingFace 镜像会导致加载失败的问题。

## 1. 数据集加载流程

### 1.1 GSM8K 任务配置

GSM8K 任务定义在 `/src/lighteval/tasks/tasks/gsm8k.py` 文件中：

```python
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",        # HuggingFace 仓库路径
    hf_subset="main",               # 数据集子集名称
    hf_avail_splits=["train", "test"],  # 可用的数据划分
    evaluation_splits=["test"],     # 用于评估的划分
    # ... 其他配置
)
```

关键参数：
- `hf_repo`: "openai/gsm8k" - 指定 HuggingFace Hub 上的数据集仓库
- `hf_subset`: "main" - 数据集的配置名称
- `hf_revision`: None（默认） - 数据集的版本，默认使用最新版本

### 1.2 数据集加载的核心代码

数据集加载的核心实现在 `/src/lighteval/tasks/lighteval_task.py` 中的 `download_dataset_worker` 方法：

```python
@staticmethod
def download_dataset_worker(
    task: "LightevalTask",
) -> DatasetDict:
    """Worker function to download a dataset from the HuggingFace Hub."""
    dataset = load_dataset(
        path=task.dataset_path,           # 例如: "openai/gsm8k"
        name=task.dataset_config_name,    # 例如: "main"
        revision=task.dataset_revision,   # 例如: None
    )
    
    if task.dataset_filter is not None:
        dataset = dataset.filter(task.dataset_filter)
    
    return dataset
```

### 1.3 调用链路

完整的加载流程如下：

1. **Pipeline 初始化** (`/src/lighteval/pipeline.py`)
   ```python
   def _init_tasks_and_requests(self, tasks: str):
       self.registry = Registry(tasks=tasks, ...)
       self.tasks_dict = self.registry.load_tasks()
       LightevalTask.load_datasets(self.tasks_dict, dataset_loading_processes)
   ```

2. **批量加载数据集** (`lighteval_task.py`)
   ```python
   @staticmethod
   def load_datasets(tasks: dict[str, "LightevalTask"], dataset_loading_processes: int = 1):
       if dataset_loading_processes <= 1:
           datasets = [task.download_dataset_worker(task) for task in tasks.values()]
       else:
           # 使用多进程并行加载
           with Pool(processes=dataset_loading_processes) as pool:
               datasets = pool.starmap(LightevalTask.download_dataset_worker, [tasks.values()])
   ```

3. **获取文档** (`lighteval_task.py`)
   ```python
   def _get_docs_from_split(self, splits: list[str], few_shots=False):
       if self.dataset is None:
           self.dataset = self.download_dataset_worker(self)
       # 从数据集中提取文档
   ```

## 2. HuggingFace 镜像问题分析

### 2.1 问题根源

**核心问题**：lighteval 使用的 `datasets.load_dataset()` 函数调用时**没有传递任何下载配置参数**，导致某些关键设置无法生效。

在 `download_dataset_worker` 方法中，只传递了三个参数：
- `path`: 数据集路径
- `name`: 数据集配置名称  
- `revision`: 数据集版本

**缺失的参数**：
- `download_config`: 下载配置对象（DownloadConfig）
- `token`: HuggingFace 访问令牌
- `trust_remote_code`: 是否信任远程代码
- `download_mode`: 下载模式（重新下载、重用缓存等）

### 2.2 为什么镜像会失败

当使用 HuggingFace 镜像站点时（通过设置环境变量 `HF_ENDPOINT`），`datasets` 库的行为可能会受到影响：

#### 问题 1: 元数据与数据文件的来源不一致

`load_dataset()` 的工作流程：
1. 首先从 HuggingFace Hub API 获取数据集元数据（dataset card, README, 配置信息等）
2. 然后下载实际的数据文件（Parquet、CSV、JSON 等）

**镜像问题**：
- 镜像站可能只同步了数据文件，但没有完整同步 API 端点
- 元数据请求可能会超时或返回不完整的信息
- 某些数据集的 loading script 可能需要访问官方 API

#### 问题 2: 缺少认证令牌传递

lighteval 在调用 `load_dataset()` 时没有传递 `token` 参数：

```python
# 当前实现
dataset = load_dataset(
    path=task.dataset_path,
    name=task.dataset_config_name,
    revision=task.dataset_revision,
    # 缺失: token=...
)
```

这导致：
- 无法访问需要认证的私有数据集
- 镜像站点的认证机制无法正常工作
- API 请求可能被限流或拒绝

#### 问题 3: 没有使用 DownloadConfig

`datasets` 库支持通过 `DownloadConfig` 对象配置下载行为：

```python
from datasets import DownloadConfig

download_config = DownloadConfig(
    cache_dir="...",           # 缓存目录
    force_download=False,      # 是否强制重新下载
    resume_download=True,      # 是否支持断点续传
    proxies={"http": "...", "https": "..."},  # 代理设置
    user_agent="...",          # 用户代理
    use_auth_token=None,       # 认证令牌
    max_retries=3,             # 最大重试次数
)
```

lighteval 没有传递这个配置，导致：
- 无法自定义下载行为
- 无法设置代理
- 无法配置重试策略
- 镜像相关的特殊设置无法应用

### 2.3 具体失败场景

使用镜像时可能遇到的错误：

1. **连接超时**
   ```
   ConnectionError: Couldn't reach https://huggingface.co/datasets/openai/gsm8k
   ```
   原因：元数据请求仍然访问官方地址，但网络不通

2. **文件下载失败**
   ```
   FileNotFoundError: Dataset 'openai/gsm8k' doesn't exist on the Hub
   ```
   原因：镜像站的 API 端点不完整

3. **认证失败**
   ```
   HfHubHTTPError: 401 Client Error: Unauthorized
   ```
   原因：镜像站需要认证但 token 未传递

## 3. 解决方案建议

### 3.1 短期方案：添加下载配置支持

修改 `download_dataset_worker` 方法，支持传递更多参数：

```python
@staticmethod
def download_dataset_worker(
    task: "LightevalTask",
    download_config: DownloadConfig | None = None,
    token: str | None = None,
) -> DatasetDict:
    """Worker function to download a dataset from the HuggingFace Hub."""
    
    # 支持从环境变量读取 token
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    dataset = load_dataset(
        path=task.dataset_path,
        name=task.dataset_config_name,
        revision=task.dataset_revision,
        download_config=download_config,  # 新增
        token=token,                      # 新增
    )
    
    if task.dataset_filter is not None:
        dataset = dataset.filter(task.dataset_filter)
    
    return dataset
```

### 3.2 中期方案：配置化下载参数

在 `LightevalTaskConfig` 中添加下载配置字段：

```python
@dataclass
class LightevalTaskConfig:
    # ... 现有字段 ...
    
    # 新增下载配置
    download_config: DownloadConfig | None = None
    use_auth_token: bool | str | None = None
    trust_remote_code: bool = False
```

### 3.3 长期方案：镜像感知的加载策略

实现智能的镜像检测和回退机制：

```python
def download_dataset_with_fallback(task, max_retries=3):
    """尝试从镜像加载，失败时回退到官方源"""
    
    # 首先尝试当前配置（可能是镜像）
    try:
        return load_dataset(...)
    except Exception as e:
        logger.warning(f"Failed to load from current endpoint: {e}")
        
        # 回退到官方源
        with temporarily_unset_hf_endpoint():
            return load_dataset(...)
```

## 4. 总结

### 问题核心

lighteval 在加载数据集时使用了最简化的 `load_dataset()` 调用，只传递了三个基本参数（path, name, revision），没有传递任何高级配置参数。这导致：

1. **无法使用 HuggingFace 镜像**：镜像站需要的特殊配置无法生效
2. **认证令牌无法传递**：私有数据集和需要认证的镜像无法访问
3. **下载行为无法自定义**：无法配置代理、缓存、重试等行为
4. **缺乏错误处理和回退机制**：遇到问题时无法自动切换到备用源

### 导致镜像失败的具体动作

- **Action 1**: 直接调用 `load_dataset(path, name, revision)` 而不传递 `download_config`
- **Action 2**: 不传递 `token` 参数，导致认证信息丢失
- **Action 3**: 不支持自定义 API 端点，完全依赖环境变量
- **Action 4**: 没有实现镜像失败时的回退机制

### 建议的改进方向

1. 添加 `download_config` 参数支持
2. 支持通过环境变量或配置文件传递 `token`
3. 实现智能的镜像检测和回退机制
4. 提供更详细的错误信息和调试日志

---

**调查日期**: 2025-12-27  
**调查人员**: GitHub Copilot  
**相关代码版本**: main branch
