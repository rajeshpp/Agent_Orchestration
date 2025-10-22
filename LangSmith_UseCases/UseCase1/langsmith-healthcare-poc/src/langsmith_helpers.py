from langsmith import Client, traceable
from typing import Optional
from .config import settings

_client: Optional[Client] = None


def get_langsmith_client() -> Client:
    """Return a singleton LangSmith client (compatible with 0.4.37)."""
    global _client
    if _client is None:
        # No project_name arg supported in this version
        _client = Client()
    return _client


def trace_fn(run_type: str = "llm", name: str = None):
    """
    Return a decorator that marks a function traceable by LangSmith.

    For langsmith<=0.4.37, project_name must be passed as metadata
    because Client() doesn't accept project_name directly.
    """
    client = get_langsmith_client()

    def _decorator(func):
        return traceable(
            run_type=run_type,
            name=name or func.__name__,
            client=client,
            metadata={"project_name": settings.PROJECT_NAME},
        )(func)

    return _decorator
