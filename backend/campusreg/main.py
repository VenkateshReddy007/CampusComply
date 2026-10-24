"""ASGI entry point. Run with ``uvicorn campusreg.main:app``."""

from campusreg.api.app import create_app

app = create_app()
