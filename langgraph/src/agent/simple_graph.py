"""A simple graph example that switches randomly between happy and sad moods."""

from typing import Literal
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
import random


class State(TypedDict):
    """State of the simple graph."""

    messages: str


def mood(state: State) -> State:
    """Mood node."""
    return {"messages": "Now, I am"}


def happy(state: State) -> State:
    """Happy node."""
    return {"messages": state["messages"] + " happy!"}


def sad(state: State) -> State:
    """Sad node."""
    return {"messages": state["messages"] + " sad!"}


def decide_mood(state: State) -> Literal["happy", "sad"]:
    """Conditional edge."""
    # Conditional edge to split 50 / 50 between happy or sad node
    threshold = 0.5
    if random.random() < threshold:
        return "happy"
    return "sad"


def dag() -> CompiledStateGraph:
    """Create and compile the state graph."""
    workflow = StateGraph(State)
    workflow.add_node("mood", mood)
    workflow.add_node("happy", happy)
    workflow.add_node("sad", sad)
    workflow.add_edge(START, "mood")
    workflow.add_conditional_edges("mood", decide_mood)
    workflow.add_edge("happy", END)
    workflow.add_edge("sad", END)

    return workflow.compile(name="Simple Graph")


try:
    simple_graph = dag()
except Exception as e:
    raise Exception(f"Error: {e}") from e
