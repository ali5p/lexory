"""Request-scoped authentication context for database row-level security."""

from contextvars import ContextVar

current_user_id_ctx: ContextVar[str | None] = ContextVar(
    "current_user_id",
    default=None,
)
