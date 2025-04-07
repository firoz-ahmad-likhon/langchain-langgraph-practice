"""Extract cv information from a PDF file."""

import time
from typing import cast, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from helper.helper import Helper
from helper.prompt import Prompt
from model.cv import CV
from dotenv import load_dotenv

load_dotenv()  # loading .env


def extract(llm_model: ChatOllama, text: str) -> dict[str, Any]:
    """Call the locally hosted llm model in Ollama to extract structured data."""
    # Define the chat prompt template
    prompt_template = ChatPromptTemplate([
        ("system", "{instruction}"),
        ("human", "PDF Content:\n{content}"),
    ])
    prompt = (prompt_template
              .with_config(run_name="Extraction Chat")
              .invoke({"instruction": Prompt.prompt_cv(),
                       "content": text}))

    try:
        response = (llm_model
                    .with_structured_output(schema=CV)
                    .with_config(run_name="Extraction")
                    .invoke(prompt))
        return cast(dict[str, Any], response.model_dump())
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}") from e


def run() -> None:
    """Process all."""
    # Define paths
    cv_folder = Path("resource/cv")
    json_folder = Path("resource/json")
    text_folder = Path("resource/text")
    # Initialize Ollama
    llm = ChatOllama(model="llama3.2",  # gemma:7b deepseek-r1:8b llama3.2 gemma3:4b
                     temperature=.1,
                     )

    for file in cv_folder.iterdir():
        if file.suffix.lower() == ".pdf":
            start_time = time.time()  # Capture start timestamp
            print("Processing: ", file.name)

            # Extract text from PDF
            content = Helper.extract_text_from_pdf(file)
            # Save raw text to file
            Helper.save_content(text_folder, file.stem, content, "txt")

            # Call Ollama
            try:
                data = extract(llm, content)
                # Save structured data to JSON
                Helper.save_content(json_folder, file.stem, data, "json")
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")

            end_time = time.time()  # Capture end timestamp
            print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
