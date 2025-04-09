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


class OllamaSingleton:
    """Singleton class to manage the Ollama model instance."""

    _instance: ChatOllama | None = None

    @classmethod
    def get_instance(cls, model_name: str = "llama3.2") -> ChatOllama:
        """Get the single instance of the ChatOllama model."""
        if cls._instance is None or cls._instance.model != model_name:
            cls._instance = ChatOllama(model=model_name, temperature=0.1)
        return cls._instance


def assistant(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Process message with LLM."""
    llm_model = OllamaSingleton.get_instance(
        config.get("configurable", {}).get("model") or "llama3.2",
    )

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

    # Use LangGraph's built-in persistence
    return workflow.compile(name="Chatbot")


try:
    chatbot = dag()
except Exception as e:
    raise RuntimeError(f"Error: {e}") from e
