"""A simple chatbot."""

from typing import Any
from functools import partial
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


def assistant(state: MessagesState, llm_model: ChatOllama) -> dict[str, Any]:
    """Process message with LLM."""
    prompt_template = ChatPromptTemplate([
        ("system", "You are a smart chatbot. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt_template.with_config(run_name="Prompt").invoke(state)
    return {"messages": llm_model.with_config(run_name="Chatting").invoke(prompt)}


def dag(llm_model: ChatOllama) -> CompiledStateGraph:
    """Create and compile the state graph."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", partial(assistant, llm_model=llm_model))
    workflow.add_edge(START, "assistant")

    # Use LangGraph's built-in persistence
    return workflow.compile(name="Chatbot")


try:
    llm = ChatOllama(model="llama3.2", temperature=.1)
    chatbot = dag(llm)
except Exception as e:
    raise RuntimeError(f"Ollama API error: {e}") from e
