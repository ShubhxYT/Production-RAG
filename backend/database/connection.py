"""Database engine and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_database_url

_engine = None
_session_factory = None


def get_engine():
    """Return a cached SQLAlchemy engine.

    Uses the DATABASE_URL from config with connection pooling.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """Return a cached sessionmaker bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory


def get_session() -> Session:
    """Create and return a new database session."""
    factory = get_session_factory()
    return factory()


def reset_engine():
    """Dispose of the current engine and reset cached state.

    Useful for testing or when reconfiguring the database URL.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
