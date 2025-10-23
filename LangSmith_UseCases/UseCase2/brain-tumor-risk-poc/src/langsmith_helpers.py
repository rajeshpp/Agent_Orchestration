from langsmith import Client, traceable
from typing import Optional
from .config import settings

_client: Optional[Client] = None


def get_langsmith_client() -> Client:
    """Create LangSmith client."""
    global _client
    if _client is None:
        _client = Client()
    return _client


def trace_fn(run_type: str = "chain", name: str = None):
    """Decorator helper for LangSmith tracing with metadata."""
    client = get_langsmith_client()

    def _decorator(func):
        return traceable(
            run_type=run_type,
            name=name or func.__name__,
            client=client,
            metadata={"project_name": settings.LANGCHAIN_PROJECT},
        )(func)

    return _decorator
