# 轮询代理:2×TP1 replica 臂用。逐请求轮转转发到多个后端,支持流式。
# 用法: python rr_proxy.py --port 8300 --backends 127.0.0.1:8101 127.0.0.1:8201
import argparse
import itertools
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clients = [
        httpx.AsyncClient(timeout=None, base_url=f"http://{b}") for b in app.state.backends
    ]
    app.state.rr = itertools.cycle(range(len(app.state.clients)))
    yield
    for c in app.state.clients:
        await c.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"backends": app.state.backends}


async def _forward(request: Request, path: str):
    client = app.state.clients[next(app.state.rr)]
    body = await request.body()
    req = client.build_request(
        "POST", path, content=body, headers={"Content-Type": "application/json"}
    )
    resp = await client.send(req, stream=True)

    async def gen():
        async for chunk in resp.aiter_raw():
            yield chunk
        await resp.aclose()

    return StreamingResponse(
        gen(), status_code=resp.status_code, media_type=resp.headers.get("content-type")
    )


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward(request, "/v1/chat/completions")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backends", nargs="+", required=True)
    args = parser.parse_args()
    app.state.backends = args.backends
    uvicorn.run(app, host=args.host, port=args.port)
