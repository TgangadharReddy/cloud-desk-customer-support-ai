"""
Convenience launcher: serves the chat UI (frontend/) and the existing
FastAPI app (app.main:app) on the SAME origin, so the browser never hits
a CORS restriction when the UI calls POST /chat.

IMPORTANT: this file does not modify app/main.py in any way. It imports
the already-fully-defined `app` object as-is and mounts a static file
directory onto that running instance. /health and /chat are registered
inside app/main.py itself (before this script ever touches `app`), so
they are matched first by FastAPI's router; the static mount only
serves paths that don't match a real API route.

Usage:
    python serve_with_ui.py

Then open:
    http://127.0.0.1:8000/        -> chat UI
    http://127.0.0.1:8000/health  -> existing health check
    http://127.0.0.1:8000/chat    -> existing chat API
    http://127.0.0.1:8000/docs    -> existing Swagger UI

Why this exists:
    Opening frontend/index.html directly as a file (or serving it from a
    separate dev server on a different port) makes the browser treat it
    as a different origin than the API. Since app/main.py intentionally
    has no CORS middleware added (out of scope for this phase), a
    cross-origin fetch to POST /chat would be blocked by the browser's
    own security policy -- not a bug in the frontend code, just how
    browsers work. Serving both from one process sidesteps that
    entirely, with zero changes to the existing backend file.
"""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.main import app

frontend_dir = Path(__file__).parent / "frontend"

# html=True makes StaticFiles serve frontend/index.html for the "/" path
# automatically, the same way a typical static site host would.
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)