from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore, SearchItem
from typing_extensions import Annotated


def retrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
) -> list[SearchItem]:
    """Retrieve a tool to use, given a search query."""
    return store.search(("tools",), query=query, limit=2)


async def aretrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
) -> list[SearchItem]:
    """Retrieve a tool to use, given a search query."""
    return await store.asearch(("tools",), query=query, limit=2)
