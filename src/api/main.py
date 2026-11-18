"""Uvicorn entry point: ``uvicorn src.api.main:app``."""
from src.api.app import app

__all__ = ["app"]
