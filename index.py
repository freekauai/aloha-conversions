"""Vercel entry point. Vercel detects FastAPI from this root module and routes
every request to `app`; locally and in Docker, run backend.main:app directly."""
import os

os.environ.setdefault("MAX_UPLOAD_MB", "4")  # Vercel's request-body limit is 4.5 MB

from backend.main import app  # noqa: E402,F401
