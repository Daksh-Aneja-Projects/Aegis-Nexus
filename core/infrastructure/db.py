from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
import logging

logger = logging.getLogger(__name__)

# Naming convention for constraints to ensure Alembic can detect changes reliably
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)

# DB Config
DATABASE_URL = os.getenv("AEGIS_DB_URL", "sqlite+aiosqlite:///./aegis_nexus.db")

# PROD: Robust Pooling
create_engine_kwargs = {
    "echo": os.getenv("AEGIS_DB_ECHO", "False") == "True",
    "pool_pre_ping": True, # Detect dead connections
}

if "sqlite" not in DATABASE_URL:
    # PostGreSQL Tuning for High Concurrency
    create_engine_kwargs.update({
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
    })

try:
    engine = create_async_engine(DATABASE_URL, **create_engine_kwargs)
    logger.info(f"✅ Database Engine Initialized: {DATABASE_URL.split('://')[0]} (Pool: {create_engine_kwargs.get('pool_size', 'Default')})")
except Exception as e:
    logger.critical(f"❌ FATAL: Potential DB Connection Failure: {e}")
    # We don't raise here to allow imports, but app startup will fail if DB needed
    engine = None

async_session_maker = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncSession:
    """Dependency for providing async session"""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
