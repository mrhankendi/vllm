# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms.power import (
    MockPowerManager,
    NvmlPowerManager,
    PowerManager,
)

pytestmark = pytest.mark.cpu_test


def test_power_manager_protocol():
    """Verify MockPowerManager and NvmlPowerManager conform to PowerManager protocol."""
    mock = MockPowerManager()
    assert isinstance(mock, PowerManager)
    nvml = NvmlPowerManager(device_ids=[0])
    assert isinstance(nvml, PowerManager)


def test_mock_power_manager_basic_get_set():
    """Verify basic set_power_cap_w and get_power_cap_w behavior on mock."""
    mock = MockPowerManager(min_power_cap_w=100, max_power_cap_w=400)
    assert mock.get_power_cap_w() is None

    mock.set_power_cap_w(200)
    assert mock.get_power_cap_w() == 200
    assert mock.set_power_cap_calls == [200]
    assert mock.get_current_power_w() == 200.0


def test_mock_power_manager_multi_device():
    """Verify multi-device per-GPU semantics on mock."""
    mock = MockPowerManager(device_ids=[0, 1, 2, 3])
    mock.set_power_cap_w(250)
    assert mock.get_power_cap_w() == 250
    assert mock._power_caps_w == {0: 250, 1: 250, 2: 250, 3: 250}

    # Inconsistent state detection
    mock._power_caps_w[1] = 200
    with pytest.raises(RuntimeError, match="Inconsistent GPU power caps"):
        mock.get_power_cap_w()


@pytest.mark.parametrize("watts", [99, 401, 0, -50])
def test_mock_power_manager_bounds_validation(watts: int):
    """Verify mock enforces min and max device limits."""
    mock = MockPowerManager(min_power_cap_w=100, max_power_cap_w=400)
    with pytest.raises(ValueError, match="outside device-supported range"):
        mock.set_power_cap_w(watts)


@pytest.mark.parametrize("invalid_val", [True, False, 150.5, "200", None])
def test_mock_power_manager_type_validation(invalid_val):
    """Verify mock rejects non-integer values."""
    mock = MockPowerManager()
    with pytest.raises(TypeError, match="Power cap must be an integer"):
        mock.set_power_cap_w(invalid_val)


def test_mock_power_manager_simulated_failure():
    """Verify failure injection on mock."""
    mock = MockPowerManager(fail_on_set=RuntimeError("Hardware failure"))
    with pytest.raises(RuntimeError, match="Hardware failure"):
        mock.set_power_cap_w(200)


def test_optional_dependency_no_nvml_init_on_import():
    """Verify importing power module on macOS/CPU does not invoke nvmlInit."""
    with patch("vllm.third_party.pynvml.nvmlInit") as mock_init:
        from vllm.platforms.power import MockPowerManager, NvmlPowerManager

        m = MockPowerManager()
        m.set_power_cap_w(200)
        _ = NvmlPowerManager(device_ids=[0])
        assert not mock_init.called


def test_nvml_power_manager_missing_library():
    """Verify clear error message when NVML library is missing."""
    nvml_mgr = NvmlPowerManager(device_ids=[0])

    from vllm.third_party.pynvml import NVMLError_LibraryNotFound

    with (
        patch(
            "vllm.third_party.pynvml.nvmlInit",
            side_effect=NVMLError_LibraryNotFound(),
        ),
        pytest.raises(RuntimeError, match="NVIDIA NVML shared library not found"),
    ):
        nvml_mgr.set_power_cap_w(200)


def test_nvml_power_manager_success_multi_gpu():
    """Verify NvmlPowerManager sets per-GPU limit in milliwatts across devices."""
    nvml_mgr = NvmlPowerManager(device_ids=[0, 1])

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"handle_{idx}"
    # Range: [150W, 350W] -> [150000 mW, 350000 mW]
    fake_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = [
        150000,
        350000,
    ]
    fake_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000

    with patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml):
        nvml_mgr.set_power_cap_w(200)

        # Verified conversion to milliwatts per device
        assert fake_pynvml.nvmlDeviceSetPowerManagementLimit.call_count == 2
        fake_pynvml.nvmlDeviceSetPowerManagementLimit.assert_any_call(
            "handle_0", 200000
        )
        fake_pynvml.nvmlDeviceSetPowerManagementLimit.assert_any_call(
            "handle_1", 200000
        )

        # Readback
        assert nvml_mgr.get_power_cap_w() == 250


def test_nvml_power_manager_out_of_range_rejected_before_mutation():
    """Verify out-of-range power cap is rejected before any device is updated."""
    nvml_mgr = NvmlPowerManager(device_ids=[0, 1])

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"handle_{idx}"
    fake_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = [
        150000,
        350000,
    ]

    with patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml):
        with pytest.raises(ValueError, match="outside device-supported range"):
            nvml_mgr.set_power_cap_w(100)

        assert not fake_pynvml.nvmlDeviceSetPowerManagementLimit.called


def test_nvml_power_manager_rollback_on_partial_failure():
    """Verify rollback is attempted if a device update fails halfway."""
    nvml_mgr = NvmlPowerManager(device_ids=[0, 1])

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"handle_{idx}"
    fake_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = [
        150000,
        350000,
    ]
    # Initial limit is 250W (250000 mW)
    fake_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000

    # Device 0 succeeds, device 1 fails
    def set_limit(handle, limit):
        if handle == "handle_1":
            raise RuntimeError("Driver bus error")

    fake_pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = set_limit

    with patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml):
        with pytest.raises(RuntimeError, match="Failed to set power cap to 200W"):
            nvml_mgr.set_power_cap_w(200)

        # Device 0 should have been rolled back to original limit (250000 mW)
        fake_pynvml.nvmlDeviceSetPowerManagementLimit.assert_any_call(
            "handle_0", 250000
        )


def test_nvml_power_manager_permission_error_message():
    """Verify clear error message on NVML permission failure."""
    nvml_mgr = NvmlPowerManager(device_ids=[0])

    from vllm.third_party.pynvml import NVMLError_NoPermission

    fake_pynvml = MagicMock()
    fake_pynvml.NVMLError_NoPermission = NVMLError_NoPermission
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
    fake_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = [
        150000,
        350000,
    ]
    fake_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000
    fake_pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = NVMLError_NoPermission()

    with (
        patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml),
        patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="Denied")),
        pytest.raises(PermissionError, match="CAP_SYS_ADMIN"),
    ):
        nvml_mgr.set_power_cap_w(200)


def test_nvml_power_manager_fallback_to_sudo_smi():
    """Verify fallback to sudo nvidia-smi succeeds when NVML gives NoPermission."""
    nvml_mgr = NvmlPowerManager(device_ids=[0])

    from vllm.third_party.pynvml import NVMLError_NoPermission

    fake_pynvml = MagicMock()
    fake_pynvml.NVMLError_NoPermission = NVMLError_NoPermission
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
    fake_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = [
        150000,
        350000,
    ]
    fake_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000
    fake_pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = NVMLError_NoPermission()
    fake_pci = MagicMock()
    fake_pci.busId = b"00000000:4A:00.0"
    fake_pynvml.nvmlDeviceGetPciInfo.return_value = fake_pci

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("vllm.utils.import_utils.import_pynvml", return_value=fake_pynvml),
        patch("subprocess.run", return_value=mock_proc) as mock_run,
    ):
        nvml_mgr.set_power_cap_w(200)

        mock_run.assert_called_once_with(
            ["sudo", "-n", "nvidia-smi", "-i", "00000000:4A:00.0", "-pl", "200"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )

