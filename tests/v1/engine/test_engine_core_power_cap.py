# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.platforms.power import MockPowerManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    EngineCoreClient,
    InprocClient,
    SyncMPClient,
    _process_utility_output,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


@pytest.fixture(params=[InprocClient, SyncMPClient, AsyncMPClient])
def power_cap_client(request):
    """Mock sockets while exercising utility serialization and engine dispatch."""
    scheduler = object.__new__(Scheduler)
    scheduler.max_num_running_reqs = 128
    scheduler.max_num_active_reqs = 16
    mock_pm = MockPowerManager(min_power_cap_w=100, max_power_cap_w=400)

    core = object.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.power_manager = mock_pm
    core.shutdown_state = EngineShutdownState.RUNNING
    core.input_queue = Queue()
    core.output_queue = Queue()
    core.has_work = Mock(return_value=True)

    client = object.__new__(request.param)
    if isinstance(client, InprocClient):
        client.engine_core = core
        return client, scheduler, mock_pm

    client.resources = SimpleNamespace(engine_dead=False)
    client.core_engine = b"engine-0"
    client.client_index = 0
    client.utility_results = {}
    client.encoder = MsgpackEncoder()
    client._ensure_output_queue_task = Mock()
    request_decoder = MsgpackDecoder()
    output_decoder = MsgpackDecoder(EngineCoreOutputs)

    def send_multipart(frames, copy):
        engine, request_type, *payload = frames
        assert engine == client.core_engine
        assert request_type == EngineCoreRequestType.UTILITY.value
        core.input_queue.put_nowait(
            (EngineCoreRequestType.UTILITY, request_decoder.decode(payload))
        )
        core._process_input_queue()
        client_index, outputs = core.output_queue.get_nowait()
        assert client_index == client.client_index
        outputs = output_decoder.decode(client.encoder.encode(outputs))
        _process_utility_output(outputs.utility_output, client.utility_results)

    socket_mock = AsyncMock if isinstance(client, AsyncMPClient) else Mock
    client.input_socket = SimpleNamespace(
        send_multipart=socket_mock(side_effect=send_multipart)
    )
    return client, scheduler, mock_pm


async def _set_power_cap(client: EngineCoreClient, watts: int) -> None:
    if isinstance(client, AsyncMPClient):
        await client.set_gpu_power_cap_w_async(watts)
    else:
        client.set_gpu_power_cap_w(watts)


async def _get_power_cap(client: EngineCoreClient) -> int | None:
    if isinstance(client, AsyncMPClient):
        return await client.get_gpu_power_cap_w_async()
    else:
        return client.get_gpu_power_cap_w()


@pytest.mark.asyncio
async def test_client_updates_gpu_power_cap(power_cap_client):
    client, scheduler, mock_pm = power_cap_client

    assert await _get_power_cap(client) is None

    await _set_power_cap(client, 200)
    assert mock_pm.get_power_cap_w() == 200
    assert await _get_power_cap(client) == 200

    # Ensure scheduler invariants remain untouched
    assert scheduler.max_num_active_reqs == 16
    assert scheduler.max_num_running_reqs == 128

    await _set_power_cap(client, 350)
    assert mock_pm.get_power_cap_w() == 350
    assert await _get_power_cap(client) == 350


@pytest.mark.asyncio
async def test_client_async_power_cap_path(power_cap_client):
    client, scheduler, mock_pm = power_cap_client

    await client.set_gpu_power_cap_w_async(250)
    assert mock_pm.get_power_cap_w() == 250
    assert await client.get_gpu_power_cap_w_async() == 250


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_cap", [50, 450])
async def test_client_propagates_invalid_power_cap(power_cap_client, invalid_cap: int):
    client, scheduler, mock_pm = power_cap_client
    error_type = ValueError if isinstance(client, InprocClient) else Exception

    with pytest.raises(error_type) as exc_info:
        await _set_power_cap(client, invalid_cap)

    assert "outside device-supported range" in str(exc_info.value)
    assert mock_pm.get_power_cap_w() is None

    # Verify scheduler was not mutated
    assert scheduler.max_num_active_reqs == 16
    assert scheduler.max_num_running_reqs == 128


@pytest.mark.asyncio
async def test_client_propagates_backend_failure(power_cap_client):
    client, scheduler, mock_pm = power_cap_client
    mock_pm.fail_on_set = RuntimeError("Driver bus communication failed")

    error_type = RuntimeError if isinstance(client, InprocClient) else Exception
    with pytest.raises(error_type) as exc_info:
        await _set_power_cap(client, 200)

    assert "Driver bus communication failed" in str(exc_info.value)
    assert mock_pm.get_power_cap_w() is None


@pytest.mark.asyncio
async def test_independent_actuators(power_cap_client):
    client, scheduler, mock_pm = power_cap_client

    # Change active limit
    if isinstance(client, AsyncMPClient):
        await client.set_max_num_active_reqs_async(64)
    else:
        client.set_max_num_active_reqs(64)

    assert scheduler.max_num_active_reqs == 64
    assert mock_pm.get_power_cap_w() is None

    # Change power cap
    await _set_power_cap(client, 300)
    assert scheduler.max_num_active_reqs == 64
    assert mock_pm.get_power_cap_w() == 300

    # Lower active limit
    if isinstance(client, AsyncMPClient):
        await client.set_max_num_active_reqs_async(32)
    else:
        client.set_max_num_active_reqs(32)

    assert scheduler.max_num_active_reqs == 32
    assert mock_pm.get_power_cap_w() == 300
