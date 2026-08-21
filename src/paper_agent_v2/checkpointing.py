from __future__ import annotations

from contextlib import nullcontext


def postgres_checkpointer(database_url: str):
    """Return a LangGraph Postgres checkpointer context for worker graph state."""
    if not database_url.startswith("postgresql"):
        return nullcontext(None)
    from langgraph.checkpoint.postgres import PostgresSaver

    connection_string = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return PostgresSaver.from_conn_string(connection_string)
