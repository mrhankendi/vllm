# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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
def active_limit_client(request):
    """Mock sockets while exercising utility serialization and engine dispatch."""
    scheduler = object.__new__(Scheduler)
    scheduler.max_num_running_reqs = 128
    scheduler.max_num_active_reqs = 16
    core = object.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.shutdown_state = EngineShutdownState.RUNNING
    core.input_queue = Queue()
    core.output_queue = Queue()
    core.has_work = Mock(return_value=True)

    client = object.__new__(request.param)
    if isinstance(client, InprocClient):
        client.engine_core = core
        return client, scheduler

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
        previous_limit = scheduler.max_num_active_reqs
        core.input_queue.put_nowait(
            (EngineCoreRequestType.UTILITY, request_decoder.decode(payload))
        )
        assert scheduler.max_num_active_reqs == previous_limit
        core._process_input_queue()
        client_index, outputs = core.output_queue.get_nowait()
        assert client_index == client.client_index
        outputs = output_decoder.decode(client.encoder.encode(outputs))
        _process_utility_output(outputs.utility_output, client.utility_results)

    socket_mock = AsyncMock if isinstance(client, AsyncMPClient) else Mock
    client.input_socket = SimpleNamespace(
        send_multipart=socket_mock(side_effect=send_multipart)
    )
    return client, scheduler


async def _set_active_limit(client: EngineCoreClient, limit: int) -> None:
    if isinstance(client, AsyncMPClient):
        await client.set_max_num_active_reqs_async(limit)
    else:
        client.set_max_num_active_reqs(limit)


@pytest.mark.asyncio
async def test_client_updates_scheduler_active_limit(active_limit_client):
    client, scheduler = active_limit_client

    await _set_active_limit(client, 64)
    assert scheduler.max_num_active_reqs == 64

    await _set_active_limit(client, 16)
    assert scheduler.max_num_active_reqs == 16
    assert scheduler.max_num_running_reqs == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 129])
async def test_client_propagates_invalid_active_limit(active_limit_client, limit: int):
    client, scheduler = active_limit_client
    message = (
        "max_num_active_reqs must be between 1 and "
        f"max_num_running_reqs (128), got {limit}"
    )
    error_type = ValueError if isinstance(client, InprocClient) else Exception

    with pytest.raises(error_type) as exc_info:
        await _set_active_limit(client, limit)

    expected = (
        message
        if isinstance(client, InprocClient)
        else f"Call to set_max_num_active_reqs method failed: {message}"
    )
    assert str(exc_info.value) == expected
    assert scheduler.max_num_active_reqs == 16

    await _set_active_limit(client, 64)
    assert scheduler.max_num_active_reqs == 64
