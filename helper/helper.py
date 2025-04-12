from pathlib import Path, PurePath
from typing import Any, cast
import json
import fitz  # type: ignore
import re
import unicodedata
from IPython.display import Image
from langchain_core.messages.base import BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class Helper:
    """Helper class for PDF text extraction."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove Unicode symbols, ASCII control characters, and icon-generated text."""
        text = unicodedata.normalize("NFKD", text)

        # Remove non-alphanumeric Unicode symbols (emoji/icons)
        text = re.sub(r"[^\w\s@,./#!?;:\"'()\-+]", " ", text)

        # Remove artifacts like "envel⌢pe" and "♂phone"
        text = re.sub(r"\b\w*[\u2000-\u206F\u2E00-\u2E7F\u25A0-\u25FF]+\w*\b", "", text)

        # Normalize spaces and line breaks
        # text = re.sub(r"\s+", " ", text).strip()
        # text = re.sub(r"\n\s*\n", "\n", text)

        return text

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from a PDF while ignoring icons."""
        doc = fitz.open(str(file_path))
        content = ""

        for page in doc:
            text = page.get_text("text")  # Extract standard text only
            if text:
                cleaned_text = Helper.clean_text(text)
                content += cleaned_text + "\n"

        return content

    @staticmethod
    def save_content(
        file_path: Path,
        file_name: str,
        text: Any,
        file_type: str = "json",
    ) -> bool:
        """Save content to a file.

        :param file_path: Directory path to save the file.
        :param file_name: Name of the file.
        :param text: Content of the file.
        :param file_type: Type [json, txt] of the file to save.
        :return: True if successful, False or Exception if file type is invalid.
        """
        file_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        file = file_path / f"{file_name}.{file_type}"  # Construct file path

        if file_type == "json":
            with open(file, "w", encoding="utf-8") as json_file:
                json.dump(text, json_file, ensure_ascii=False, indent=4)  # Save as JSON
        elif file_type == "txt":
            with open(file, "w", encoding="utf-8") as txt_file:
                txt_file.write(str(text))  # Save as plain text

        elif file_type in ["png", "img"]:
            # Handle both IPython Image or raw bytes
            if isinstance(text, Image):
                img_data = text.data
            elif isinstance(text, bytes):
                img_data = text
            else:
                raise ValueError(
                    "Invalid image content. Must be bytes or IPython.display.Image.",
                )

            with open(file, "wb") as img_file:
                img_file.write(img_data)

        else:
            raise ValueError("Invalid file type. Use 'json', 'txt', or 'png'.")

        return True

    @staticmethod
    def parse_pdf(pdf_path: PurePath) -> list[Document]:
        """Parse the PDF data."""
        loader = PyPDFLoader(pdf_path)
        return cast(list[Document], loader.load())

    @staticmethod
    def accumulate_agent_messages(response: dict[str, list[BaseMessage]]) -> str:
        """Accumulates and returns all message contents from the agent response.

        :param response: The response from the LangGraph agent.
        Returns: Concatenated message contents.
        """
        messages = response.get("messages", [])
        collected_output = []

        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                collected_output.append(str(msg.content))

        return "\n".join(collected_output)
