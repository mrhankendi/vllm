# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import math
import os
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

from vllm.logger import init_logger
from vllm.third_party.pynvml import (
    NVMLError_LibraryNotFound,
    NVMLError_NoPermission,
    NVMLError_NotSupported,
)

logger = init_logger(__name__)


@runtime_checkable
class PowerManager(Protocol):
    """Hardware abstraction for GPU power management."""

    def set_power_cap_w(self, watts: int) -> None:
        """Set the per-GPU power cap in watts across all managed GPUs.

        Args:
            watts: Target power limit in watts per GPU.

        Raises:
            TypeError: If watts is not an integer.
            ValueError: If watts is non-positive or outside device limits.
            RuntimeError: If setting power cap fails on any device.

        """
        ...

    def get_power_cap_w(self) -> int | None:
        """Get the configured per-GPU power cap in watts.

        Returns:
            The uniform per-GPU power cap in watts, or None if unconfigured.

        Raises:
            RuntimeError: If managed GPUs have inconsistent power caps.

        """
        ...

    def get_current_power_w(self) -> float | None:
        """Get the instantaneous power draw in watts, if supported."""
        ...

    def close(self) -> None:
        """Release underlying management resources, if any."""
        ...


class MockPowerManager:
    """Mock power manager for CPU-only and macOS unit tests."""

    def __init__(
        self,
        current_power_cap_w: int | None = None,
        min_power_cap_w: int = 100,
        max_power_cap_w: int = 400,
        device_ids: Sequence[int] | None = None,
        fail_on_set: Exception | None = None,
    ):
        self.device_ids = list(device_ids) if device_ids is not None else [0]
        self.min_power_cap_w = min_power_cap_w
        self.max_power_cap_w = max_power_cap_w
        self.fail_on_set = fail_on_set
        self._power_caps_w: dict[int, int | None] = {
            dev_id: current_power_cap_w for dev_id in self.device_ids
        }
        self.set_power_cap_calls: list[int] = []

    def set_power_cap_w(self, watts: int) -> None:
        if isinstance(watts, bool) or not isinstance(watts, int):
            raise TypeError(f"Power cap must be an integer, got {type(watts).__name__}")
        if self.fail_on_set is not None:
            raise self.fail_on_set
        if not (self.min_power_cap_w <= watts <= self.max_power_cap_w):
            raise ValueError(
                f"Requested power cap {watts}W is outside device-supported "
                f"range [{self.min_power_cap_w}W, {self.max_power_cap_w}W]"
            )
        self.set_power_cap_calls.append(watts)
        for dev_id in self.device_ids:
            self._power_caps_w[dev_id] = watts

    def get_power_cap_w(self) -> int | None:
        caps = set(self._power_caps_w.values())
        if not caps or (None in caps and len(caps) == 1):
            return None
        if len(caps) > 1:
            raise RuntimeError(f"Inconsistent GPU power caps: {self._power_caps_w}")
        return next(iter(caps))

    def get_current_power_w(self) -> float | None:
        cap = self.get_power_cap_w()
        return float(cap) if cap is not None else None

    def close(self) -> None:
        pass


class NvmlPowerManager:
    """NVIDIA NVML-backed power manager for real GPU hardware.

    Applies per-GPU power caps using the vendored pynvml library.
    Provides atomic multi-GPU application with automatic rollback on partial
    failures.
    """

    def __init__(
        self,
        device_ids: Sequence[int] | None = None,
    ):
        self._explicit_device_ids = list(device_ids) if device_ids is not None else None
        self._resolved_device_ids: list[int] | None = None
        self._is_closed = False

    @contextlib.contextmanager
    def _nvml_session(self):
        """Context manager managing an active NVML session."""
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        try:
            pynvml.nvmlInit()
        except NVMLError_LibraryNotFound as e:
            raise RuntimeError(
                "NVIDIA NVML shared library not found. GPU power management "
                "requires NVIDIA drivers and NVML."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}") from e

        try:
            yield pynvml
        finally:
            with suppress(Exception):
                pynvml.nvmlShutdown()

    def _get_device_ids(self, pynvml: Any) -> list[int]:
        if self._explicit_device_ids is not None:
            return self._explicit_device_ids

        if self._resolved_device_ids is not None:
            return self._resolved_device_ids

        from vllm.platforms.interface import get_assigned_physical_gpu_ids

        assigned = get_assigned_physical_gpu_ids()
        if assigned is not None:
            self._resolved_device_ids = list(assigned)
            return self._resolved_device_ids

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            from vllm.platforms import current_platform

            ids = []
            for token in cuda_visible.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    phys_id = current_platform.device_control_id_to_physical_device_id(
                        token
                    )
                    ids.append(phys_id)
                except Exception:
                    logger.warning(
                        "Unable to map CUDA_VISIBLE_DEVICES token %r to "
                        "physical device ID",
                        token,
                    )
            if ids:
                self._resolved_device_ids = ids
                return self._resolved_device_ids

        count = pynvml.nvmlDeviceGetCount()
        self._resolved_device_ids = list(range(count))
        return self._resolved_device_ids

    def _get_handles(self, pynvml: Any) -> dict[int, Any]:
        device_ids = self._get_device_ids(pynvml)
        if not device_ids:
            raise RuntimeError("No GPU devices available for power management")
        handles = {}
        for dev_id in device_ids:
            try:
                handles[dev_id] = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to obtain NVML handle for GPU {dev_id}: {e}"
                ) from e
        return handles

    def set_power_cap_w(self, watts: int) -> None:
        if isinstance(watts, bool) or not isinstance(watts, int):
            raise TypeError(f"Power cap must be an integer, got {type(watts).__name__}")
        if watts <= 0:
            raise ValueError(f"Power cap must be positive, got {watts}W")

        watts_mw = watts * 1000

        with self._nvml_session() as pynvml:
            handles = self._get_handles(pynvml)

            # 1. Validate constraints across all target devices before any mutation
            for dev_id, handle in handles.items():
                try:
                    min_mw, max_mw = (
                        pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                    )
                except NVMLError_NotSupported as e:
                    raise RuntimeError(
                        f"Power management is not supported on GPU {dev_id} "
                        "(e.g., MIG instance or unsupported architecture)"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to query power management constraints for "
                        f"GPU {dev_id}: {e}"
                    ) from e

                min_w = math.ceil(min_mw / 1000.0)
                max_w = max_mw // 1000
                if not (min_mw <= watts_mw <= max_mw):
                    raise ValueError(
                        f"Requested power cap {watts}W is outside device-supported "
                        f"range [{min_w}W, {max_w}W] for GPU {dev_id}"
                    )

            # 2. Query previous power limits for rollback support
            previous_limits_mw: dict[int, int] = {}
            for dev_id, handle in handles.items():
                try:
                    prev_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    previous_limits_mw[dev_id] = prev_mw
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to read current power limit on GPU {dev_id} "
                        f"prior to update: {e}"
                    ) from e

            # 3. Apply the per-GPU power cap sequentially with rollback on failure
            updated_devices: list[int] = []
            for dev_id, handle in handles.items():
                try:
                    pynvml.nvmlDeviceSetPowerManagementLimit(handle, watts_mw)
                    updated_devices.append(dev_id)
                except NVMLError_NoPermission as e:
                    rollback_err = self._rollback(
                        pynvml, handles, updated_devices, previous_limits_mw
                    )
                    msg = (
                        f"Failed to set power cap to {watts}W on GPU {dev_id}: "
                        "NVML permission denied (requires root or "
                        "CAP_SYS_ADMIN privileges)."
                    )
                    if rollback_err:
                        msg += f" Rollback error: {rollback_err}"
                    raise PermissionError(msg) from e
                except Exception as e:
                    rollback_err = self._rollback(
                        pynvml, handles, updated_devices, previous_limits_mw
                    )
                    msg = f"Failed to set power cap to {watts}W on GPU {dev_id}: {e}."
                    if rollback_err:
                        msg += f" Rollback error: {rollback_err}"
                    raise RuntimeError(msg) from e

    def _rollback(
        self,
        pynvml: Any,
        handles: dict[int, Any],
        updated_devices: list[int],
        previous_limits_mw: dict[int, int],
    ) -> str | None:
        rollback_failures = []
        for dev_id in updated_devices:
            orig_mw = previous_limits_mw.get(dev_id)
            if orig_mw is None:
                continue
            try:
                pynvml.nvmlDeviceSetPowerManagementLimit(handles[dev_id], orig_mw)
            except Exception as e:
                rollback_failures.append(f"GPU {dev_id}: {e}")
        if rollback_failures:
            return (
                "Rollback failed on: "
                + "; ".join(rollback_failures)
                + ". GPUs may have inconsistent power states."
            )
        return None

    def get_power_cap_w(self) -> int | None:
        with self._nvml_session() as pynvml:
            handles = self._get_handles(pynvml)
            caps_w: dict[int, int] = {}
            for dev_id, handle in handles.items():
                try:
                    limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    caps_w[dev_id] = round(limit_mw / 1000.0)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to read power cap on GPU {dev_id}: {e}"
                    ) from e

            unique_caps = set(caps_w.values())
            if not unique_caps:
                return None
            if len(unique_caps) > 1:
                raise RuntimeError(
                    f"Inconsistent GPU power caps across devices: {caps_w}"
                )
            return next(iter(unique_caps))

    def get_current_power_w(self) -> float | None:
        with self._nvml_session() as pynvml:
            handles = self._get_handles(pynvml)
            usages = []
            for dev_id, handle in handles.items():
                try:
                    usage_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    usages.append(usage_mw / 1000.0)
                except Exception as e:
                    logger.debug(
                        "Unable to read instantaneous power on GPU %d: %s",
                        dev_id,
                        e,
                    )
            return sum(usages) / len(usages) if usages else None

    def close(self) -> None:
        self._is_closed = True
