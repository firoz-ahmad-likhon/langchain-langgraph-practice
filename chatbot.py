"""A simple chatbot with memory."""

from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from langsmith import traceable
from IPython.display import Image
from helper.helper import Helper
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loading .env


def assistant(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Call the llm model."""
    llm_model = config.get("configurable", {}).get("model")
    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "You talk like a pirate. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
    prompt = prompt_template.with_config(run_name="Prompt").invoke(state)

    return {"messages": llm_model.with_config(run_name="Chatting").invoke(prompt)}


def dag() -> CompiledStateGraph:
    """Create the state graph."""
    workflow = StateGraph(state_schema=MessagesState)  # New graph
    workflow.add_edge(START, "assistant")  # Single node in the graph
    workflow.add_node("assistant", assistant)
    memory = MemorySaver()  # Add memory
    return workflow.compile(checkpointer=memory, name="Chatbot")


@traceable(name="Simple Chatbot")
def run() -> Any:
    """Process all."""
    try:
        # Initialize Ollama
        model = ChatOllama(
            model="llama3.2",
            temperature=0.1,
        )
        app = dag()  # Create state graph and compile it
        img = Image(app.get_graph().draw_mermaid_png())  # Generate graph image
        Helper.save_content(Path("resource/image"), "dag_chatbot", img, "png")
        # Set config for the dag
        config = {"configurable": {"thread_id": "chat123", "model": model}}

        input_messages = [HumanMessage("Hi! I'm Bob.")]
        output_1 = app.invoke({"messages": input_messages}, config)

        input_messages = [HumanMessage("What is my name?")]
        output_2 = app.invoke({"messages": input_messages}, config)

        return output_1 + output_2
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e


if __name__ == "__main__":
    run()
