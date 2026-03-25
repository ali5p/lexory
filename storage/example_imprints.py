"""
Example imprint storage: SQLite + SQLAlchemy.
Stores chronological imprints of example points (mistake_examples) for fallback retrieval.
"""
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class ExampleImprint(Base):
    """Imprint of an example point payload for chronological lookup."""

    __tablename__ = "example_imprints"

    mistake_id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(index=True)
    session_id: Mapped[str] = mapped_column(index=True)
    timestamp: Mapped[str] = mapped_column(index=True)  # ISO format for ordering
    # Optional fields (nullable)
    user_text_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    rule_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    mistake_type: Mapped[Optional[str]] = mapped_column(nullable=True)


class ExampleImprintStore:
    """
    Persistent store for example point imprints.
    Authoritative source for chronological mistake retrieval.
    """

    def __init__(self, db_path: str = "./lexory_imprints.db"):
        path = Path(db_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(
            uri,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )

    def insert(self, payload: dict) -> None:
        """Insert an imprint from an example point payload."""
        required = ("mistake_id", "user_id", "session_id", "timestamp")
        for k in required:
            if k not in payload:
                raise ValueError(f"Example imprint requires '{k}' in payload")
        with self._session_factory() as session:
            imprint = ExampleImprint(
                mistake_id=payload["mistake_id"],
                user_id=payload["user_id"],
                session_id=payload["session_id"],
                timestamp=payload["timestamp"],
                user_text_id=payload.get("user_text_id"),
                rule_id=payload.get("rule_id"),
                mistake_type=payload.get("mistake_type"),
            )
            session.add(imprint)
            try:
                session.commit()
            except SQLAlchemyError:
                session.rollback()
                raise

    def get_most_recent_mistake_id(self, user_id: str) -> Optional[str]:
        """Get the most recent mistake_id for a user (chronological)."""
        with self._session_factory() as session:
            stmt = (
                select(ExampleImprint.mistake_id)
                .where(ExampleImprint.user_id == user_id)
                .order_by(ExampleImprint.timestamp.desc())
                .limit(1)
            )
            row = session.execute(stmt).first()
            return row[0] if row else None
