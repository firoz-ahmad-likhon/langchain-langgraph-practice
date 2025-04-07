"""A simple chatbot with memory."""

import time
from typing import Any
from functools import partial
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image
from helper.helper import Helper
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loading .env


def assistant(state: MessagesState, llm_model: ChatOllama) -> dict[str, Any]:
    """Call the llm model."""
    # Define the chat prompt template
    prompt_template = ChatPromptTemplate([
        ("system", "You talk like a pirate. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt_template.with_config(run_name="Prompt").invoke(state)

    return {"messages": llm_model.with_config(run_name="Chatting").invoke(prompt)}


def app_workflow(llm_model: ChatOllama) -> CompiledStateGraph:
    """Create the state graph."""
    workflow = StateGraph(state_schema=MessagesState)  # New graph
    workflow.add_edge(START, "assistant")  # Single node in the graph
    workflow.add_node("assistant", partial(assistant, llm_model=llm_model))
    memory = MemorySaver()  # Add memory
    return workflow.compile(checkpointer=memory, name="Chatbot")


def run() -> None:
    """Process all."""
    start_time = time.time()  # Capture start timestamp
    print("Chatting...")

    try:
        # Initialize Ollama
        model = ChatOllama(model="llama3.2",  # gemma:7b deepseek-r1:8b llama3.2 gemma3:4b
                           temperature=.1,
                           )
        app = app_workflow(model)  # Create state graph and compile it
        # View
        img = Image(app.get_graph().draw_mermaid_png())
        Helper.save_content(Path("resource/image"), "dag_chatbot", img, "png")
        # Set config for the graph
        config = {"configurable": {"thread_id": "abc123"}}

        input_messages = [HumanMessage("Hi! I'm Bob.")]
        output = app.invoke({"messages": input_messages}, config)
        # output contains all messages in state
        output["messages"][-1].pretty_print()

        input_messages = [HumanMessage("What is my name?")]
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e

    end_time = time.time()  # Capture end timestamp
    print(f"Chatting time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
