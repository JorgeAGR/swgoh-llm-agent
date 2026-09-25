"""FastAPI app serving the chat UI and a streaming chat endpoint."""

import json
import logging
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from swgoh_agent.agent.graph import create_agent
from swgoh_agent.config import Settings, get_settings

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=64)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = create_agent(settings)
        yield

    app = FastAPI(title="SWGOH Agent", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return files("swgoh_agent.web").joinpath("static/index.html").read_text()

    @app.get("/api/info")
    async def info(request: Request) -> dict:
        catalog = request.app.state.agent.catalog
        return {
            "model": settings.llm_model,
            "units": len(catalog.units),
            "fetched_at": catalog.dataset.fetched_at,
        }

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        agent = request.app.state.agent

        async def events():
            try:
                async for event in agent.astream(body.message, body.thread_id):
                    yield _sse(event)
            except Exception as exc:
                log.exception("Chat turn failed")
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
