"""
Database connection and session management

Provides:
- Async database engine
- Session factory
- Database initialization
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from backend.config.settings import settings
from backend.database.models import Base


# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    poolclass=NullPool,  # Use NullPool for async to avoid connection issues
    future=True
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def init_db():
    """
    Initialize database

    Creates all tables defined in models
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db():
    """
    Drop all database tables

    WARNING: This will delete all data!
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def get_session() -> AsyncSession:
    """
    Get database session

    Yields:
        AsyncSession instance
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
