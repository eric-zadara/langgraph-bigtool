import math
import types
import uuid

from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models import GenericFakeChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import convert_positional_only_function_to_tool

EMBEDDING_SIZE = 1536


# Create a list of all the functions in the math module
all_names = dir(math)

math_functions = [
    getattr(math, name)
    for name in all_names
    if isinstance(getattr(math, name), types.BuiltinFunctionType)
]

# Convert to tools, handling positional-only arguments (idiosyncrasy of math module)
all_tools = []
for function in math_functions:
    if wrapper := convert_positional_only_function_to_tool(function):
        all_tools.append(wrapper)

# Store tool objects in registry
tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}

fake_embeddings = DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, *args, **kwargs) -> "FakeModel":
        """Do nothing for now."""
        return self


acos_tool = next(tool for tool in tool_registry.values() if tool.name == "acos")
initial_query = f"{acos_tool.name}: {acos_tool.description}"  # make same as embedding
fake_llm = FakeModel(
    messages=iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "retrieve_tools",
                        "args": {"query": initial_query},
                        "id": "abc123",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "acos",
                        "args": {"x": 0.5},
                        "id": "abc234",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("The arc cosine of 0.5 is approximately 1.047 radians."),
        ]
    )
)


def run_end_to_end_test(llm: LanguageModelLike, embeddings: Embeddings) -> None:
    # Store tool descriptions in store
    store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": EMBEDDING_SIZE,
            "fields": ["description"],
        }
    )
    for tool_id, tool in tool_registry.items():
        store.put(
            ("tools",),
            tool_id,
            {
                "description": f"{tool.name}: {tool.description}",
            },
        )

    builder = create_agent(llm, tool_registry)
    agent = builder.compile(store=store)

    result = agent.invoke(
        {"messages": "Use available tools to calculate arc cosine of 0.5."}
    )
    assert set(result.keys()) == {"messages", "selected_tool_ids"}
    assert "acos" in [
        tool_registry[tool_id].name for tool_id in result["selected_tool_ids"]
    ]
    assert set(message.type for message in result["messages"]) == {
        "human",
        "ai",
        "tool",
    }
    tool_calls = [
        tool_call
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    assert tool_calls
    tool_call_names = [tool_call["name"] for tool_call in tool_calls]
    assert "retrieve_tools" in tool_call_names
    math_tool_calls = [
        tool_call for tool_call in tool_calls if tool_call["name"] == "acos"
    ]
    assert len(math_tool_calls) == 1
    math_tool_call = math_tool_calls[0]
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id == math_tool_call["id"]
    ]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert round(float(tool_message.content), 4) == 1.0472
    reply = result["messages"][-1]
    assert isinstance(reply, AIMessage)
    assert not reply.tool_calls
    assert reply.content


def test_end_to_end() -> None:
    run_end_to_end_test(fake_llm, fake_embeddings)
