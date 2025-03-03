from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing_extensions import Annotated


def retrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
):
    """Retrieve a tool to use, given a search query."""
    return store.search(("tools",), query=query, limit=2)


def aretrieve_tools(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore],
):
    """Retrieve a tool to use, given a search query."""
    return store.asearch(("tools",), query=query, limit=2)
