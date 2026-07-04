"""Database connection management.

Provides a single get_engine() factory so all modules use the same engine
instance. Defaults to SQLite for the demo; the interface is identical for
PostGIS (swap the connection URL and you're done).

Usage:
    from src.database.connection import get_engine
    from src.database.models import init_db

    init_db("data/ada.db")
    engine = get_engine("data/da.db")
    # engine.execute(...)
"""
from __future__ import annotations

import pathlib
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event


@lru_cache(maxsize=1)
def get_engine(db_path: str = "data/ada.db") -> Engine:
    """Return a cached SQLAlchemy Engine for the given SQLite database.

    The engine is cached so that all callers share the same connection pool
    (important for SQLite which has single-writer semantics).

    Args:
        db_path: Path to the SQLite database file, relative to the project
                 root, or absolute.

    Returns:
        A SQLAlchemy Engine instance.

    To switch to PostGIS later, change the URL prefix:
        engine = create_engine(f"postgresql+psycopg2://user:pass@host:5432/{db_name}")
    """
    # Resolve relative to the project root (2 levels up from this file)
    resolved = pathlib.Path(db_path)
    if not resolved.is_absolute():
        resolved = pathlib.Path(__file__).resolve().parents[2] / db_path
    resolved.parent.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{resolved.as_posix()}"
    engine = create_engine(db_url, echo=False, future=True)

    # Enable WAL mode + foreign keys for SQLite (safe defaults)
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine
