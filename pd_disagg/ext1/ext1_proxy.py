# SPDX-License-Identifier: Apache-2.0
# EXT1 instrumented 1P1D pull proxy, derived from smoke/toy_proxy_v0251.py.
# Changes vs toy proxy:
#   1. Honors incoming X-Request-Id header (falls back to uuid4) so the client
#      controls request identity end-to-end (client -> proxy -> P -> D).
#   2. Emits one "EXT1_PROXY" JSON line per request with host-epoch timestamps:
#      t_recv / t_p_send / t_p_done / t_d_send / t_first_chunk / t_last_chunk.
# Single-host deployment => all epochs share one clock domain with the
# EXT1_KV lines logged by the patched D-side connector.

import argparse
import itertools
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )
    yield
    for c in app.state.prefill_clients + app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--prefiller-hosts", "--prefiller-host", type=str, nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("prefiller hosts/ports mismatch")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("decoder hosts/ports mismatch")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_next_client(app, service_type: str):
    if service_type == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    if service_type == "decode":
        return app.state.decode_clients[next(app.state.decode_iterator)]
    raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await client_info["client"].post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    await response.aread()
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens
    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        t_recv = time.time()
        req_data = await request.json()
        # EXT1: honor client-supplied identity
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        timing = {"request_id": request_id, "t_recv": t_recv}

        prefill_client_info = get_next_client(request.app, "prefill")
        timing["t_p_send"] = time.time()
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )
        timing["t_p_done"] = time.time()

        response_json = response.json()
        await response.aclose()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        decode_client_info = get_next_client(request.app, "decode")

        async def generate_stream():
            first = True
            timing["t_d_send"] = time.time()
            async for chunk in stream_service_response(
                decode_client_info, api, req_data, request_id=request_id
            ):
                if first:
                    timing["t_first_chunk"] = time.time()
                    first = False
                yield chunk
            timing["t_last_chunk"] = time.time()
            print("EXT1_PROXY " + json.dumps(timing), flush=True)

        return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
