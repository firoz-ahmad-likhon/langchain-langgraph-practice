"""Convert InfluxDB data to ChromaDB and perform semantic search on it.

Semantic search is best working for unstructured data like text. Here, we are using
InfluxDB just for practicing.

Here RunnableLambda is used to chain the functions and make it easy to read.
"""

from typing import Any
from influxdb_client_3 import InfluxDBClient3
from state.influxdb_pandas import DataState
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langsmith import traceable
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # loading .env


def extract(state: DataState) -> DataState:
    """Read data from InfluxDB."""
    query = "select * from attendance"
    try:
        reader = state.client.query(query=query, language="sql")  # Execute the query
        state.df = reader.to_pandas()  # Convert the result to a pandas DataFrame
        return state
    except Exception as e:
        raise Exception("An error occurred: ", e) from e


def transform(state: DataState, embedding_model: OllamaEmbeddings) -> DataState:
    """Transform the data."""
    try:
        # Generate unique IDs for each document
        state.df["unique_id"] = state.df.apply(
            lambda row: f"{row['time']}_{row['name']}_{row['present']}",
            axis=1,
        )

        # Generate structured page content
        state.df["page_content"] = state.df.apply(
            lambda row: f"At {row['time']},  {row['name']} was {'present' if row['present'] else 'absent'}.",
            axis=1,
        )

        # Generate embeddings
        state.df["vector"] = embedding_model.embed_documents(
            state.df["page_content"].tolist(),
        )
        return state
    except Exception as e:
        raise Exception(f"Error during processing: {e}") from e


def load(state: DataState, vector_store: Chroma) -> int:
    """Store the data in ChromaDB."""
    try:
        # Create Document objects for each row
        documents = [
            Document(
                page_content=row["page_content"],
                metadata={
                    col: (
                        str(row[col])
                        if isinstance(row[col], pd.Timestamp)
                        else row[col]
                    )
                    for col in state.df.columns
                    if col not in ["page_content", "vector", "unique_id"]
                },
                id=row["unique_id"],
            )
            for _, row in state.df.iterrows()
        ]

        vector_store.add_documents(documents)
        return len(state.df)
    except Exception as e:
        raise Exception(
            f"Error occurred while storing embeddings in ChromaDB: {str(e)}",
        ) from e


@traceable(name="Query Test", run_type="retriever")
def query_test(vector_store: Chroma) -> dict[str, Any]:
    """Query the data."""
    # Get the count of documents
    documents_ = vector_store.get()

    # Semantic search
    results = vector_store.similarity_search(
        "Give me the unique students who were absent at any days",
    )
    return {
        "documents_length": len(documents_["ids"]),
        "results": results,
    }


@traceable(name="InfluxDB to ChromaDB")
def run() -> str:
    """Process all."""
    # Initialize the client
    client = InfluxDBClient3(
        token=os.getenv("INFLUXDB_TOKEN"),
        host=os.getenv("INFLUXDB_HOST"),
        org=os.getenv("INFLUXDB_ORG"),
        database=os.getenv("INFLUXDB_DB"),
    )

    # Initialize Ollama embeddings model
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_URL"),
    )

    # Initialize LangChain Chroma vector store
    vector_store = Chroma(
        collection_name="students",
        embedding_function=embedding_model,
        persist_directory=os.getenv("CHROMADB_LOC"),
    )

    # Workflow
    chain = (
        RunnableLambda(lambda state: extract(state), name="Extract")
        | RunnableLambda(
            lambda state: transform(state, embedding_model),
            name="Transform",
        )
        | RunnableLambda(lambda state: load(state, vector_store), name="Load")
    )

    chain.name = "ETL"
    chain.invoke(DataState(client=client))
    # RunnableSequence(chain, name="InfluxDB to ChromaDB").invoke(DataState(client=client))

    # Query test
    testing = RunnableLambda(lambda _: query_test(vector_store), name="Query Test")
    testing.name = "testing"
    testing.invoke(None)

    return "Done!"


if __name__ == "__main__":
    run()
