"""Tools for the calculator."""

import time
from typing import Any, cast
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
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


@chain
def get_tool_call(llm_model: ChatOllama, tools: list[BaseTool]) -> list[dict[Any, Any]]:
    """Create a tool call."""
    # Define the chat prompt template
    prompt_template = ChatPromptTemplate([
        ("system", "{instruction}"),
        ("human", "Question:\n{question}"),
    ])
    return cast(list[dict[Any, Any]], llm_model.bind_tools(tools, tool_choice="any").invoke(prompt_template.invoke(
        {"instruction": "You are an expert calculator.",
         "question": "What is 3 * 12? Also, what is 11 + 49?"})).tool_calls)


@chain
def get_tool_message(tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
    """Execute the tools and return the tool calls and responses."""
    messages = {}
    for tool_call in tool_calls:
        selected_tool = {"addition-tool": add,
                         "multiplication-tool": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages[tool_call["name"]] = tool_msg

    return messages


@chain
def feed_back(messages: dict[str, str], llm_model: ChatOllama) -> BaseMessage:
    """Create a final response."""
    prompt_template = ChatPromptTemplate([
        ("system", "You are an expert calculator."),
        ("human", "Question: What is 3 * 12? Also, what is 11 + 49?"),
        ("system", "You are a helpful assistant."),
        ("human", "Display the tool response in natural language."),
        ("human",
         "Here are the results of the calculations:\n{results}\nPlease provide a final answer."),
    ])
    return llm_model.invoke(prompt_template.invoke({"results": messages}))


def run() -> None:
    """Process all."""
    start_time = time.time()  # Capture start timestamp
    print("Processing...")
    # Inspect the tool.
    print(multiply.name, multiply.description,
          multiply.args, multiply.return_direct)

    try:
        # Initialize Ollama
        llm = ChatOllama(model="llama3.2",  # gemma:7b deepseek-r1:8b llama3.2 gemma3:4b
                         temperature=.1,
                         )
        chain = (
            get_tool_call.bind(tools=[multiply, add]).with_config(
                run_name="Get tool calls")
            | get_tool_message.with_config(run_name="Read messages")
            | feed_back.bind(llm_model=llm).with_config(run_name="Feed model back")
        )
        chain.name = "Calculator"
        chain.invoke(llm)

    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e

    end_time = time.time()  # Capture end timestamp
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
