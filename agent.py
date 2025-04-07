"""A calculator agent."""

import time
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from helper.helper import Helper
from dotenv import load_dotenv

load_dotenv()  # loading .env


class CalculatorInput(BaseModel):
    """Input for multiplication tool."""

    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True, response_format="content_and_artifact", parse_docstring=True)
def multiply(a: int, b: int) -> tuple[str, int]:
    """Multiply two numbers."""
    result = a * b
    return f"The product of {a} and {b} is {result}.", result


@tool("addition-tool", args_schema=CalculatorInput, return_direct=True, response_format="content_and_artifact", parse_docstring=True)
def add(a: int, b: int) -> tuple[str, int]:
    """Add two numbers."""
    result = a + b
    return f"The sum of {a} and {b} is {result}.", result


@tool("default-tool", return_direct=True)
def fallback_tool(query: str) -> str:
    """Fallback tool for general questions."""
    return f"I'm a calculator agent and can only help with math questions like addition and multiplication. You asked: '{query}'"


def run() -> None:
    """Process all."""
    start_time = time.time()  # Capture start timestamp
    print("Processing...")

    try:
        tools = [multiply, add, fallback_tool]

        memory = MemorySaver()
        model = ChatOllama(model="llama3.2",  temperature=.1).bind_tools(tools)

        # # Create the agent executor
        agent_executor = create_react_agent(
            model, tools, checkpointer=memory, name="Calculator Agent")
        config = {"configurable": {"thread_id": "abc1234"}}
        response = agent_executor.invoke({"messages": [HumanMessage(
            content="What is 3 * 12? Also, what is 11 + 49? What is your name?")]}, config)
        output = Helper.accumulate_agent_messages(response)
        print(output)
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e

    end_time = time.time()  # Capture end timestamp
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
