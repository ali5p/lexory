"""Request-scoped authentication context for database row-level security."""

from contextvars import ContextVar

current_user_id_ctx: ContextVar[str | None] = ContextVar(
    "current_user_id",
    default=None,
)


def require_current_user_id() -> str:
    """Return the authenticated tenant ID or fail closed outside a request context."""
    user_id = current_user_id_ctx.get()
    if not user_id:
        raise RuntimeError("An authenticated user context is required")
    return user_id
