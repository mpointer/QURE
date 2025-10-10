"""
FastAPI application entry point

QURE - Question-Reason-Update reconciliation system
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.database.database import init_db
from backend.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events
    """
    # Startup
    print("🚀 Starting QURE API...")
    print(f"📊 Database: {settings.database_url}")
    print(f"🤖 LLM Provider: {settings.llm_provider.value}")

    # Initialize database
    await init_db()
    print("✅ Database initialized")

    yield

    # Shutdown
    print("👋 Shutting down QURE API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Question-Reason-Update reconciliation system with multi-agent LLM architecture",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": "Question-Reason-Update reconciliation system",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "cases": "/api/cases",
            "statistics": "/api/statistics"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
