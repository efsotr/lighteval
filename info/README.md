# LightEval Dataset Loading Investigation

## Overview

This folder contains the investigation results on how LightEval loads datasets and why HuggingFace mirror sites cause loading failures.

## Files

### 1. `dataset_loading_analysis_zh.md` (Chinese)
Complete analysis of the dataset loading mechanism in Chinese, including:
- Detailed code flow for GSM8K dataset loading
- Root cause analysis of mirror failures
- Recommended solutions

### 2. `dataset_loading_analysis_en.md` (English)
English version of the complete analysis, covering:
- GSM8K task configuration and loading process
- Why HuggingFace mirrors fail
- Short-term, mid-term, and long-term solutions

### 3. `dataset_loading_flow_diagram.md`
Visual flowcharts showing:
- Normal loading flow (official HuggingFace Hub)
- Problematic flow (with mirrors)
- Missing pieces in current implementation
- Key insights about why mirrors fail

## Quick Summary

### Problem Statement
LightEval cannot load datasets when using HuggingFace mirror sites (configured via `HF_ENDPOINT` environment variable). It only works with the official HuggingFace Hub.

### Root Cause
The `download_dataset_worker()` method in `/src/lighteval/tasks/lighteval_task.py` calls `load_dataset()` with only 3 parameters:
```python
dataset = load_dataset(
    path=task.dataset_path,
    name=task.dataset_config_name,
    revision=task.dataset_revision,
)
```

**Missing critical parameters:**
- `download_config`: For configuring download behavior (proxies, retries, cache)
- `token`: For authentication with mirrors that require it
- No fallback mechanism when loading fails

### Why This Causes Mirror Failures

1. **Metadata vs. Data Inconsistency**
   - Mirrors may not implement full HuggingFace Hub API
   - Metadata requests may fail or timeout
   - Cannot configure alternative endpoints

2. **Authentication Issues**
   - Token not passed to `load_dataset()`
   - Mirror authentication mechanisms don't work
   - Cannot access private datasets

3. **No Download Configuration**
   - Cannot set custom proxies
   - Cannot configure retry strategies
   - Cannot control cache behavior
   - Mirror-specific settings cannot be applied

### Specific Actions Causing the Problem

1. ❌ Calling `load_dataset()` without `download_config` parameter
2. ❌ Not passing `token` parameter for authentication
3. ❌ No custom error handling for mirror-specific failures
4. ❌ No fallback to official source when mirror fails
5. ❌ Fully relying on environment variables without explicit configuration

## Recommendations

### Immediate Fix
Add support for `download_config` and `token` parameters:
```python
@staticmethod
def download_dataset_worker(
    task: "LightevalTask",
    download_config: DownloadConfig | None = None,
    token: str | None = None,
) -> DatasetDict:
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    dataset = load_dataset(
        path=task.dataset_path,
        name=task.dataset_config_name,
        revision=task.dataset_revision,
        download_config=download_config,
        token=token,
    )
    return dataset
```

### Configuration Enhancement
Add download settings to `LightevalTaskConfig`:
```python
@dataclass
class LightevalTaskConfig:
    # ... existing fields ...
    download_config: DownloadConfig | None = None
    use_auth_token: bool | str | None = None
```

### Robust Solution
Implement fallback mechanism:
```python
def download_dataset_with_fallback(task, max_retries=3):
    try:
        # Try with current configuration (possibly mirror)
        return load_dataset(...)
    except Exception as e:
        logger.warning(f"Mirror failed: {e}, falling back to official source")
        # Fallback to official HuggingFace Hub
        with temporarily_unset_hf_endpoint():
            return load_dataset(...)
```

## Example: GSM8K Dataset

The investigation uses GSM8K as a concrete example:
- **Dataset**: `openai/gsm8k`
- **Configuration**: `main`
- **Task Definition**: `/src/lighteval/tasks/tasks/gsm8k.py`
- **Loading Code**: `/src/lighteval/tasks/lighteval_task.py:download_dataset_worker()`

## Investigation Details

For complete technical analysis, see:
- `dataset_loading_analysis_en.md` - Detailed English analysis
- `dataset_loading_analysis_zh.md` - Detailed Chinese analysis
- `dataset_loading_flow_diagram.md` - Visual flow diagrams

---

**Investigation Date**: December 27, 2025  
**Investigator**: GitHub Copilot  
**Repository**: efsotr/lighteval  
**Branch**: main
