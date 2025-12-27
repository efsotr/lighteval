# LightEval Dataset Loading Mechanism Analysis - Using GSM8K as an Example

## Investigation Background

This document analyzes how lighteval loads datasets, using gsm8k as an example, and investigates why using HuggingFace mirrors causes loading failures.

## 1. Dataset Loading Process

### 1.1 GSM8K Task Configuration

The GSM8K task is defined in `/src/lighteval/tasks/tasks/gsm8k.py`:

```python
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",        # HuggingFace repository path
    hf_subset="main",               # Dataset subset name
    hf_avail_splits=["train", "test"],  # Available splits
    evaluation_splits=["test"],     # Splits used for evaluation
    # ... other configurations
)
```

Key parameters:
- `hf_repo`: "openai/gsm8k" - Specifies the dataset repository on HuggingFace Hub
- `hf_subset`: "main" - Dataset configuration name
- `hf_revision`: None (default) - Dataset version, defaults to latest

### 1.2 Core Dataset Loading Code

The core implementation for dataset loading is in the `download_dataset_worker` method in `/src/lighteval/tasks/lighteval_task.py`:

```python
@staticmethod
def download_dataset_worker(
    task: "LightevalTask",
) -> DatasetDict:
    """Worker function to download a dataset from the HuggingFace Hub."""
    dataset = load_dataset(
        path=task.dataset_path,           # e.g.: "openai/gsm8k"
        name=task.dataset_config_name,    # e.g.: "main"
        revision=task.dataset_revision,   # e.g.: None
    )
    
    if task.dataset_filter is not None:
        dataset = dataset.filter(task.dataset_filter)
    
    return dataset
```

### 1.3 Call Chain

Complete loading flow:

1. **Pipeline Initialization** (`/src/lighteval/pipeline.py`)
   ```python
   def _init_tasks_and_requests(self, tasks: str):
       self.registry = Registry(tasks=tasks, ...)
       self.tasks_dict = self.registry.load_tasks()
       LightevalTask.load_datasets(self.tasks_dict, dataset_loading_processes)
   ```

2. **Batch Dataset Loading** (`lighteval_task.py`)
   ```python
   @staticmethod
   def load_datasets(tasks: dict[str, "LightevalTask"], dataset_loading_processes: int = 1):
       if dataset_loading_processes <= 1:
           datasets = [task.download_dataset_worker(task) for task in tasks.values()]
       else:
           # Use multiprocessing for parallel loading
           with Pool(processes=dataset_loading_processes) as pool:
               datasets = pool.starmap(LightevalTask.download_dataset_worker, [tasks.values()])
   ```

3. **Get Documents** (`lighteval_task.py`)
   ```python
   def _get_docs_from_split(self, splits: list[str], few_shots=False):
       if self.dataset is None:
           self.dataset = self.download_dataset_worker(self)
       # Extract documents from dataset
   ```

## 2. HuggingFace Mirror Issue Analysis

### 2.1 Root Cause

**Core Issue**: The `datasets.load_dataset()` function used by lighteval **does not pass any download configuration parameters**, preventing certain critical settings from taking effect.

In the `download_dataset_worker` method, only three parameters are passed:
- `path`: Dataset path
- `name`: Dataset configuration name
- `revision`: Dataset version

**Missing Parameters**:
- `download_config`: Download configuration object (DownloadConfig)
- `token`: HuggingFace access token
- `trust_remote_code`: Whether to trust remote code
- `download_mode`: Download mode (redownload, reuse cache, etc.)

### 2.2 Why Mirrors Fail

When using HuggingFace mirror sites (by setting the `HF_ENDPOINT` environment variable), the behavior of the `datasets` library can be affected:

#### Issue 1: Inconsistency Between Metadata and Data File Sources

`load_dataset()` workflow:
1. First fetches dataset metadata from HuggingFace Hub API (dataset card, README, configuration info, etc.)
2. Then downloads actual data files (Parquet, CSV, JSON, etc.)

**Mirror Problem**:
- Mirror sites may only sync data files but not fully sync API endpoints
- Metadata requests may timeout or return incomplete information
- Some datasets' loading scripts may require access to the official API

#### Issue 2: Missing Authentication Token Passing

Lighteval doesn't pass the `token` parameter when calling `load_dataset()`:

```python
# Current implementation
dataset = load_dataset(
    path=task.dataset_path,
    name=task.dataset_config_name,
    revision=task.dataset_revision,
    # Missing: token=...
)
```

This causes:
- Cannot access private datasets requiring authentication
- Mirror site authentication mechanisms don't work properly
- API requests may be rate-limited or rejected

#### Issue 3: No DownloadConfig Usage

The `datasets` library supports configuring download behavior through `DownloadConfig` objects:

```python
from datasets import DownloadConfig

download_config = DownloadConfig(
    cache_dir="...",           # Cache directory
    force_download=False,      # Whether to force redownload
    resume_download=True,      # Whether to support resuming downloads
    proxies={"http": "...", "https": "..."},  # Proxy settings
    user_agent="...",          # User agent
    use_auth_token=None,       # Authentication token
    max_retries=3,             # Maximum retry attempts
)
```

Lighteval doesn't pass this configuration, resulting in:
- Cannot customize download behavior
- Cannot set proxies
- Cannot configure retry strategies
- Mirror-related special settings cannot be applied

### 2.3 Specific Failure Scenarios

Errors that may occur when using mirrors:

1. **Connection Timeout**
   ```
   ConnectionError: Couldn't reach https://huggingface.co/datasets/openai/gsm8k
   ```
   Reason: Metadata requests still access the official address, but the network is unreachable

2. **File Download Failure**
   ```
   FileNotFoundError: Dataset 'openai/gsm8k' doesn't exist on the Hub
   ```
   Reason: Mirror site's API endpoints are incomplete

3. **Authentication Failure**
   ```
   HfHubHTTPError: 401 Client Error: Unauthorized
   ```
   Reason: Mirror site requires authentication but token wasn't passed

## 3. Solution Recommendations

### 3.1 Short-term Solution: Add Download Configuration Support

Modify the `download_dataset_worker` method to support passing more parameters:

```python
@staticmethod
def download_dataset_worker(
    task: "LightevalTask",
    download_config: DownloadConfig | None = None,
    token: str | None = None,
) -> DatasetDict:
    """Worker function to download a dataset from the HuggingFace Hub."""
    
    # Support reading token from environment variables
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    dataset = load_dataset(
        path=task.dataset_path,
        name=task.dataset_config_name,
        revision=task.dataset_revision,
        download_config=download_config,  # New
        token=token,                      # New
    )
    
    if task.dataset_filter is not None:
        dataset = dataset.filter(task.dataset_filter)
    
    return dataset
```

### 3.2 Mid-term Solution: Configurable Download Parameters

Add download configuration fields to `LightevalTaskConfig`:

```python
@dataclass
class LightevalTaskConfig:
    # ... existing fields ...
    
    # New download configuration
    download_config: DownloadConfig | None = None
    use_auth_token: bool | str | None = None
    trust_remote_code: bool = False
```

### 3.3 Long-term Solution: Mirror-aware Loading Strategy

Implement intelligent mirror detection and fallback mechanism:

```python
def download_dataset_with_fallback(task, max_retries=3):
    """Try loading from mirror, fallback to official source on failure"""
    
    # First try current configuration (possibly a mirror)
    try:
        return load_dataset(...)
    except Exception as e:
        logger.warning(f"Failed to load from current endpoint: {e}")
        
        # Fallback to official source
        with temporarily_unset_hf_endpoint():
            return load_dataset(...)
```

## 4. Summary

### Core Problem

Lighteval uses a minimalist `load_dataset()` call when loading datasets, passing only three basic parameters (path, name, revision) without any advanced configuration parameters. This results in:

1. **Cannot use HuggingFace mirrors**: Special configurations needed by mirror sites cannot take effect
2. **Authentication tokens cannot be passed**: Private datasets and mirrors requiring authentication cannot be accessed
3. **Download behavior cannot be customized**: Cannot configure proxies, cache, retries, etc.
4. **Lack of error handling and fallback mechanisms**: Cannot automatically switch to backup sources when encountering problems

### Specific Actions That Cause Mirror Failures

- **Action 1**: Directly calling `load_dataset(path, name, revision)` without passing `download_config`
- **Action 2**: Not passing the `token` parameter, causing authentication information to be lost
- **Action 3**: Not supporting custom API endpoints, fully relying on environment variables
- **Action 4**: Not implementing fallback mechanisms when mirrors fail

### Recommended Improvements

1. Add `download_config` parameter support
2. Support passing `token` via environment variables or configuration files
3. Implement intelligent mirror detection and fallback mechanisms
4. Provide more detailed error messages and debugging logs

---

**Investigation Date**: 2025-12-27  
**Investigator**: GitHub Copilot  
**Related Code Version**: main branch
