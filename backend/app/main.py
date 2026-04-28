"""
FastAPI entrypoint for the DocAI RAG backend.
"""

# Use the OS trust store so corporate SSL inspection certs are honored
# when calling external APIs (e.g. Groq). Must run before httpx is imported.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocAI RAG API",
        version="0.1.0",
        description="Upload PDFs and ask questions over them using Groq LLMs.",
    )

    origins = [o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    @app.get("/")
    def root():
        return {
            "name": "DocAI RAG API",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
