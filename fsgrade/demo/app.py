"""
The FastAPI application.

``docs_url`` is disabled on purpose: Swagger UI pulls its JavaScript and fonts
from a CDN, which would be the one outbound request left in an otherwise
offline app. The OpenAPI JSON is still served, since it is generated locally.
"""

from __future__ import annotations

import pathlib
from typing import Any

from fsgrade.demo.settings import DemoSettings

STATIC_DIR = pathlib.Path(__file__).parent / "static"


def create_app(settings: DemoSettings | None = None) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from fsgrade.demo.api import build_router
    from fsgrade.demo.errors import DemoError

    settings = settings or DemoSettings.from_env()

    app = FastAPI(
        title="Cross-Species Quality Grading — demo",
        version="1.0.0",
        docs_url=None,          # Swagger UI would fetch from a CDN
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    @app.exception_handler(DemoError)
    async def _demo_error(_request, exc: DemoError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.include_router(build_router(settings), prefix="/api")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def _index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
