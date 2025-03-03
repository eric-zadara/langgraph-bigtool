from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing_extensions import Annotated

ToolId = str


def retrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
) -> list[ToolId]:
    """Retrieve a tool to use, given a search query."""
    results = store.search(("tools",), query=query, limit=2)
    return [result.key for result in results]


async def aretrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
) -> list[ToolId]:
    """Retrieve a tool to use, given a search query."""
    results = await store.asearch(("tools",), query=query, limit=2)
    return [result.key for result in results]
