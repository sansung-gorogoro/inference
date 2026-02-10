from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("LXP5 Inference Service starting up")
    yield
    logger.info("LXP5 Inference Service shutting down")


app = FastAPI(
    title="LXP5 Inference Service",
    description="STT -> RAG Index -> MCQ Quiz Generation Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def root_health_check() -> HealthResponse:
    return HealthResponse(status="ok")


app.include_router(router)
