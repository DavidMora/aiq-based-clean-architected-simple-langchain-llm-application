"""Main application entry point."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.presentation.api.routes import router


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="AIQ Clean Architecture Chat API",
        description="Simple LangChain LLM | Parser with clean architecture",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
