"""A router example that chooses tool call or natural response."""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from model.calculator import CalculatorInput

llm = ChatOllama(model="llama3-groq-tool-use", temperature=0.1)


@tool(
    "multiplication-tool",
    args_schema=CalculatorInput,
    return_direct=True,
    response_format="content_and_artifact",
    parse_docstring=True,
)
def multiply(a: int, b: int) -> tuple[str, int]:
    """Multiply two numbers."""
    result = a * b
    return f"The product of {a} and {b} is {result}.", result


def assistant(state: MessagesState) -> MessagesState:
    """Call tools dynamically."""
    return {"messages": [llm.bind_tools([multiply]).invoke(state["messages"])]}


def dag() -> CompiledStateGraph:
    """Create and compile the state graph."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode([multiply]))
    workflow.add_edge(START, "assistant")
    # If there is no tool to call, go directly to the end.
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", END)

    return workflow.compile(name="Router")


try:
    router = dag()
except Exception as e:
    raise Exception(f"Error: {e}") from e
