"""Append-only JSON event ledger; SQLite locally, PostgreSQL in Compose."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, select
from sqlalchemy.exc import IntegrityError


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, url: str):
        if url.startswith("sqlite:///") and ":memory:" not in url:
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                from sqlalchemy.pool import StaticPool
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        metadata = MetaData()
        self.events = Table("events", metadata,
                            Column("id", Integer, primary_key=True),
                            Column("kind", String(40), nullable=False, index=True),
                            Column("key", String(200), unique=True, nullable=False),
                            Column("created_at", String(50), nullable=False),
                            Column("payload", Text, nullable=False))
        metadata.create_all(self.engine)

    def append(self, kind: str, key: str, payload: dict) -> bool:
        # Reject NaN rather than writing JSON that downstream clients cannot parse.
        encoded = json.dumps(payload, allow_nan=False, default=str)
        try:
            with self.engine.begin() as conn:
                conn.execute(self.events.insert().values(kind=kind, key=key,
                                                        created_at=utcnow(), payload=encoded))
            return True
        except IntegrityError:
            return False

    def list(self, kind: str, limit: int = 100, ascending: bool = False) -> list[dict]:
        order = self.events.c.id.asc() if ascending else self.events.c.id.desc()
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.events).where(self.events.c.kind == kind)
                                .order_by(order).limit(limit)).mappings().all()
        return [{**json.loads(r["payload"]), "ledger_sequence": r["id"]} for r in rows]

    def latest(self, kind: str) -> dict | None:
        rows = self.list(kind, 1)
        return rows[0] if rows else None

