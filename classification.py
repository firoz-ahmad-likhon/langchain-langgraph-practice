"""Identify the sentiment of a given text."""

import time
from pathlib import Path
from typing import cast
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from helper.helper import Helper
from helper.prompt import Prompt
from model.classification import Classification
from dotenv import load_dotenv

load_dotenv()  # loading .env


def classify(llm_model: ChatOllama, text: str) -> dict[str, str]:
    """Call the locally hosted llm model in Ollama to extract structured data."""
    # Define the chat prompt template
    prompt_template = ChatPromptTemplate([
        ("system", "{instruction}"),
        ("human", "Passage:\n{content}"),
    ])
    prompt = (prompt_template
              .with_config(run_name="Classification Chat")
              .invoke({"instruction": Prompt.prompt_classification(),
                       "content": text}))

    try:
        response = (llm_model
                    .with_structured_output(schema=Classification)
                    .with_config(run_name="Classification")
                    .invoke(prompt))
        return cast(dict[str, str], response.model_dump())
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e


def run() -> None:
    """Process all."""
    start_time = time.time()  # Capture start timestamp
    print("Processing...")
    try:
        # Initialize Ollama
        llm = ChatOllama(model="llama3.2",  # gemma:7b deepseek-r1:8b llama3.2 gemma3:4b
                         temperature=.1,
                         )
        data = classify(llm,
                        "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!")
        # Save structured data to JSON
        Helper.save_content(Path("resource/json"),
                            "classification", data, "json")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e

    end_time = time.time()  # Capture end timestamp
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
