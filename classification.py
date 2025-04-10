"""Identify the sentiment of a given text."""

from pathlib import Path
from typing import cast
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from helper.helper import Helper
from helper.prompt import Prompt
from model.classification import Classification
from dotenv import load_dotenv

load_dotenv()  # loading .env


def classify(llm_model: ChatOllama, text: str) -> dict[str, str]:
    """Call the locally hosted llm model in Ollama to extract structured data."""
    # Define the chat prompt template
    prompt_template = ChatPromptTemplate(
        [
            ("system", "{instruction}"),
            ("human", "Passage:\n{content}"),
        ],
    )
    prompt = prompt_template.with_config(run_name="Prompt").invoke(
        {"instruction": Prompt.prompt_classification(), "content": text},
    )

    try:
        response = (
            llm_model.with_structured_output(schema=Classification)
            .with_config(run_name="Classifying")
            .invoke(prompt)
        )
        return cast(dict[str, str], response.model_dump())
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e


@traceable(name="Sentiment Classification")
def run() -> str:
    """Process all."""
    try:
        # Initialize Ollama
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,
        )
        data = classify(
            llm,
            "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!",
        )
        # Save structured data to JSON
        Helper.save_content(Path("resource/json"), "classification", data, "json")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e

    return "Sentiment classification completed."


if __name__ == "__main__":
    run()
