# Dataset Loading Flow Diagram

## Normal Flow (Official HuggingFace Hub)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Initialization                       │
│                 (_init_tasks_and_requests)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Registry.load_tasks()                         │
│            Creates LightevalTask instances from                  │
│                  task configurations                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              LightevalTask.load_datasets()                       │
│       Batch loads datasets (sequential or parallel)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            download_dataset_worker(task)                         │
│                                                                   │
│    dataset = load_dataset(                                       │
│        path="openai/gsm8k",      ✓ Passed                       │
│        name="main",              ✓ Passed                       │
│        revision=None,            ✓ Passed                       │
│        # download_config=None,   ✗ NOT Passed                   │
│        # token=None,             ✗ NOT Passed                   │
│    )                                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              datasets.load_dataset()                             │
│                                                                   │
│  1. Fetch metadata from HuggingFace Hub API                     │
│     └─> https://huggingface.co/api/datasets/openai/gsm8k       │
│                                                                   │
│  2. Download data files                                          │
│     └─> https://huggingface.co/datasets/openai/gsm8k/...       │
│                                                                   │
│  ✓ SUCCESS: Both metadata and data files accessible             │
└─────────────────────────────────────────────────────────────────┘
```

## Problematic Flow (With HuggingFace Mirror)

```
┌─────────────────────────────────────────────────────────────────┐
│            User sets: HF_ENDPOINT=https://mirror.com            │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              LightevalTask.load_datasets()                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            download_dataset_worker(task)                         │
│                                                                   │
│    dataset = load_dataset(                                       │
│        path="openai/gsm8k",                                      │
│        name="main",                                              │
│        revision=None,                                            │
│        # Missing critical parameters!                            │
│    )                                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              datasets.load_dataset()                             │
│                                                                   │
│  1. Try to fetch metadata                                        │
│     ┌─────────────────────────────────────────────────┐        │
│     │  Without download_config:                        │        │
│     │  - Cannot specify custom proxy                   │        │
│     │  - Cannot configure retry logic                  │        │
│     │  - Cannot control cache behavior                 │        │
│     └─────────────────────────────────────────────────┘        │
│     │                                                             │
│     ├─> Attempt 1: https://mirror.com/api/...                   │
│     │   ✗ FAIL: API endpoint not available on mirror            │
│     │                                                             │
│     └─> Attempt 2: https://huggingface.co/api/...               │
│         ✗ FAIL: Network unreachable (reason for using mirror)   │
│                                                                   │
│  ❌ FAILURE: ConnectionError or TimeoutError                    │
└─────────────────────────────────────────────────────────────────┘

Alternative Scenario:
┌─────────────────────────────────────────────────────────────────┐
│              datasets.load_dataset()                             │
│                                                                   │
│  1. Metadata fetch succeeds (mirror has API)                    │
│     └─> https://mirror.com/api/datasets/openai/gsm8k           │
│                                                                   │
│  2. Try to download data files                                   │
│     ┌─────────────────────────────────────────────────┐        │
│     │  Without token parameter:                        │        │
│     │  - Mirror requires authentication                │        │
│     │  - Token from env not automatically used         │        │
│     │  - No way to pass custom headers                 │        │
│     └─────────────────────────────────────────────────┘        │
│     │                                                             │
│     └─> https://mirror.com/datasets/openai/gsm8k/data.parquet  │
│         ✗ FAIL: 401 Unauthorized                                │
│                                                                   │
│  ❌ FAILURE: Authentication Error                               │
└─────────────────────────────────────────────────────────────────┘
```

## The Missing Pieces

### What lighteval currently does:
```python
load_dataset(
    path="openai/gsm8k",
    name="main",
    revision=None,
)
```

### What it should do to support mirrors:
```python
from datasets import DownloadConfig
import os

# Get token from environment
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# Create download config
download_config = DownloadConfig(
    token=token,                    # For authentication
    max_retries=3,                  # Retry on failure
    resume_download=True,           # Resume interrupted downloads
    force_download=False,           # Use cache when available
)

# Load with proper configuration
load_dataset(
    path="openai/gsm8k",
    name="main",
    revision=None,
    download_config=download_config,  # ← This is missing!
    token=token,                      # ← This is missing!
)
```

## Key Insights

### Why only official source works:
1. **Official Hub**: Metadata API and data files are both accessible
2. **No authentication needed**: Public datasets work without tokens
3. **Reliable network**: Direct connection to huggingface.co

### Why mirrors fail:
1. **Incomplete API**: Mirror may not implement full HuggingFace Hub API
2. **Authentication issues**: Token not passed, mirror auth fails
3. **Network configuration**: Cannot specify custom proxy/retry logic
4. **Mixed sources**: May try to fetch from both mirror and official (inconsistent)

### The critical missing actions:
- ❌ Not passing `download_config` parameter
- ❌ Not passing `token` parameter  
- ❌ No fallback mechanism when initial load fails
- ❌ No custom error handling for mirror-specific issues
- ❌ No way to configure download behavior per environment
