"""A simple chatbot.

The default model is llama3.2. You can change it from langgraph studio (New Assistants -> model).
"""

from typing import Any
from typing_extensions import TypedDict
from langchain_core.runnables.config import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


class ConfigSchema(TypedDict):
    """Configuration schema for the chatbot."""

    model: str | None

# Available models
models = {
    "ollama": ChatOllama(model="llama3.2", temperature=0.1)
}

def assistant(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Process message with LLM."""
    # Select model
    model_name = config.get("configurable", {}).get("model", "ollama")
    llm_model = models.get(model_name, models["ollama"])

    # Check if the user wants to quit
    last_user_msg = state["messages"][-1].content.strip().lower()
    if last_user_msg in {"quit", "exit", "q"}:
        return {"messages": [AIMessage(content="Goodbye! 👋")]}

    prompt_template = ChatPromptTemplate(
        [
            SystemMessage(
                content="You are a smart chatbot. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
    prompt = prompt_template.with_config(run_name="Prompt").invoke(state)
    return {"messages": llm_model.with_config(run_name="Chatting").invoke(prompt)}


def dag() -> CompiledStateGraph:
    """Create and compile the state graph."""
    workflow = StateGraph(MessagesState, ConfigSchema)
    workflow.add_node("assistant", assistant)
    workflow.add_edge(START, "assistant")

    return workflow.compile(name="Chatbot")


try:
    chatbot = dag()
except Exception as e:
    raise RuntimeError(f"Error: {e}") from e
