from typing import Any

from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing_extensions import Annotated

ToolId = str


def get_default_retrieval_tool(
    namespace_prefix: tuple[str, ...],
    *,
    limit: int = 2,
    filter: dict[str, Any] | None = None,
):
    """Get default sync and async functions for tool retrieval."""

    def retrieve_tools(
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ) -> list[ToolId]:
        """Retrieve a tool to use, given a search query."""
        results = store.search(
            namespace_prefix,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [result.key for result in results]

    async def aretrieve_tools(
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ) -> list[ToolId]:
        """Retrieve a tool to use, given a search query."""
        results = await store.asearch(
            namespace_prefix,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [result.key for result in results]

    return retrieve_tools, aretrieve_tools
