from typing_extensions import TypedDict


class State(TypedDict, total=False):
    """State of the rag."""

    question: str
    query: str
    result: str
    answer: str
