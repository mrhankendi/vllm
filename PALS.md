# PALS: Power Capping & Dynamic Admission Limit for vLLM v1

This document describes the PALS extension for vLLM v1, including the architecture, setup instructions for GPU/HPC clusters (e.g., Boston University SCC), and testing workflows.

---

## 1. Overview of Changes

PALS adds two runtime actuators to the vLLM v1 engine and scheduler:

1. **GPU Power Capping (`PowerManager`)**
   - Abstraction in [`vllm/platforms/power.py`](file:///usr2/postdoc/hankendi/dev/vllm/vllm/platforms/power.py) conforming to the `PowerManager` protocol.
   - **`NvmlPowerManager`**: Communicates with hardware via NVML (`pynvml`) to get/set per-GPU power caps in watts (`set_power_cap_w`, `get_power_cap_w`).
   - **`MockPowerManager`**: Simulated mock device for CPU/macOS tests and mock scenarios.
   - Actuation hooks exposed on `EngineCore` and all `EngineCoreClient` implementations (`InprocClient`, `SyncMPClient`, `AsyncMPClient`).

2. **Dynamic Request Admission Limiting (`set_max_num_active_reqs`)**
   - Runtime throttle on the Scheduler without rebuilding static KV-cache runner capacity.
   - When lowered, currently `RUNNING` requests drain normally while new requests wait until capacity drops below the limit.
   - Supported across in-process and multiprocessing engine architectures (sync and async).

---

## 2. Shared Cluster Setup Guide (HPC / SCC)

When running on an HPC cluster with prebuilt modules (e.g. `module load python3/3.13.8 cuda/13.2`) and an existing working vLLM virtual environment, you **do not need to recompile or reinstall C++/CUDA kernels**.

### Prerequisites
- Active Python environment with PyTorch and baseline vLLM (e.g., `vllm-env`).
- Test dependencies: `pip install pytest tblib pytest-asyncio`

### Fast Setup (Zero Recompilation)
The `pals` branch is 100% Python code. It reuses the precompiled `.so` binaries from your existing environment:

```bash
# 1. Load modules and activate environment
module load python3/3.13.8 cuda/13.2
source /scratch/$USER/vllm-env/bin/activate

# 2. Checkout the pals branch in your git repo
cd /usr2/postdoc/hankendi/dev/vllm
git checkout pals

# 3. Copy compiled binaries and version file from site-packages to your git repo
SITE_VLLM=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cp $SITE_VLLM/*.so /usr2/postdoc/hankendi/dev/vllm/vllm/
cp $SITE_VLLM/_version.py /usr2/postdoc/hankendi/dev/vllm/vllm/

# 4. Copy sub-module binaries (e.g. FlashAttention extensions)
cd $SITE_VLLM
find . -name "*.so" -exec cp --parents {} /usr2/postdoc/hankendi/dev/vllm/vllm/ \;
cd ~
```

*(Note: Git automatically ignores `*.so` binaries via `.gitignore`).*

---

## 3. Switching Between PALS & Default Experiments

To switch between the `pals` experimental code and your untouched baseline vLLM:

### To run PALS experiments:
```bash
export PYTHONPATH=/usr2/postdoc/hankendi/dev/vllm:$PYTHONPATH

# Sanity check
python -c "import vllm; print('Loaded from:', vllm.__file__); from vllm.platforms.power import MockPowerManager; print('Pals ready!')"
```

### To run Default baseline experiments:
```bash
unset PYTHONPATH

# Sanity check
python -c "import vllm; print('Loaded from:', vllm.__file__)"
```
Your baseline `site-packages/vllm` installation remains 100% untouched.

---

## 4. Running the Test Suite

Run the test suites with `python -m pytest` so the active environment's Python and packages are used:

```bash
# 1. Power Manager unit tests (19 tests)
python -m pytest /usr2/postdoc/hankendi/dev/vllm/tests/platforms/test_power.py -v

# 2. Engine Core GPU Power Capping tests (18 tests)
python -m pytest /usr2/postdoc/hankendi/dev/vllm/tests/v1/engine/test_engine_core_power_cap.py -v

# 3. Engine Core Admission Limit tests (12 tests)
python -m pytest /usr2/postdoc/hankendi/dev/vllm/tests/v1/engine/test_engine_core_active_limit.py -v

# 4. Scheduler Admission Limit tests (13 tests)
python -m pytest /usr2/postdoc/hankendi/dev/vllm/tests/v1/core/test_scheduler.py -v -k "active_limit"
```

All 62 tests should pass.

---

## 5. Python API Usage Examples

### Dynamic Admission Limit
```python
# Inproc engine / scheduler:
engine_client.set_max_num_active_reqs(32)

# Async engine / client:
await async_engine_client.set_max_num_active_reqs_async(16)
```

### GPU Power Capping
```python
# Inproc / sync client:
engine_client.set_gpu_power_cap_w(250)
current_cap = engine_client.get_gpu_power_cap_w()
print(f"Current power cap: {current_cap}W")

# Async client:
await async_engine_client.set_gpu_power_cap_w_async(300)
cap = await async_engine_client.get_gpu_power_cap_w_async()
```

> [!NOTE]
> Setting actual hardware power caps via NVML at runtime may require administrative or root privileges (`nvidia-smi -pl <watts>`) on the GPU host, depending on driver configuration.
