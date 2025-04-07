"""Convert the pdf data to ChromaDB and perform semantic search on it.

Here @chain decorator is used to chain the functions and make it easy to read.
"""

from typing import cast
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import PurePath
from langchain_core.runnables import chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()  # loading .env


def parse_pdf(pdf_path: PurePath) -> list[Document]:
    """Parse the PDF data."""
    loader = PyPDFLoader(pdf_path)
    return cast(list[Document], loader.load())


@chain
def extract(pdf_path: PurePath) -> list[Document]:
    """Convert PDF to documents."""
    return parse_pdf(pdf_path)


@chain
def transform(docs: list[Document]) -> list[Document]:
    """Transform documents to embeddings."""
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0, separators=["\n\n", "\n"], add_start_index=True,
    )
    return cast(list[Document], text_splitter.split_documents(docs))


@chain
def load(docs: list[Document], vector_store: Chroma) -> bool:
    """Store the data in ChromaDB."""
    try:
        vector_store.add_documents(documents=docs)
        print(f"Stored {len(docs)} documents in ChromaDB.")
        return True
    except Exception as e:
        raise Exception(
            f"Error occurred while storing embeddings in ChromaDB: {str(e)}") from e


def query_test(vector_store: Chroma) -> None:
    """Query the data."""
    # Get the count of documents
    documents_ = vector_store.get()
    print("Total number of documents in the vector store: ",
          len(documents_['ids']))

    # Semantic search
    results = vector_store.similarity_search(
        "When Bangladesh got independence?",
    )
    print(results)


@chain
def retriever(query: str, vector_store: Chroma, k: int = 10) -> list[Document]:
    """Retrieve documents from the vector store based on the query."""
    return cast(list[Document], vector_store.similarity_search(query, k=k))


@chain
def mmr_retriever(query: str, vector_store: Chroma, k: int = 10, fetch_k: int = 20, lambda_mult: float = 0.5) -> list[Document]:
    """Retrieve documents using Maximal Marginal Relevance (MMR) from the vector store."""
    return cast(
        list[Document],
        vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult),
    )


@chain
def ensemble_retriever(query: str, vector_store: Chroma, docs: list[Document]) -> list[Document]:
    """Retrieve documents using Ensemble Retriever from the vector store."""
    # Initialize BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    # Configure vector store retriever
    vector_store_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10},
    )

    # Create ensemble retriever
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store_retriever],
        weights=[0.5, 0.5],
    )

    return cast(list[Document], ensemble.invoke(query))


def run() -> None:
    """Process all."""
    start_time = time.time()  # Capture start timestamp
    print("Processing...")
    # Initialize Ollama embeddings model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text",
                                       base_url="http://localhost:11434")

    # Initialize LangChain Chroma vector store
    vector_store = Chroma(
        collection_name="bangladesh_overview",
        embedding_function=embedding_model,
        persist_directory="./chromadb",
    )

    chain = (
        extract.with_config(run_name="Extract")
        | transform.with_config(run_name="Transform")
        | load.bind(vector_store=vector_store).with_config(run_name="Load")
    )
    chain.name = "PDF to ChromaDB"
    chain.invoke(PurePath(
        "./resource/Bangladesh Overview.pdf"))

    query_test(vector_store)  # Query the data
    # simple retriever
    retriever.bind(vector_store=vector_store).batch([
        "When Bangladesh got independence?",
        "What is the history of Bangladesh?",
    ])
    # MMR Retriever
    mmr_retriever.bind(vector_store=vector_store).batch([
        "When Bangladesh got independence?",
        "What is the history of Bangladesh?",
    ])
    # Ensemble Retriever
    ensemble_retriever.bind(
        vector_store=vector_store,
        docs=parse_pdf(PurePath("./resource/Bangladesh Overview.pdf")),
    ).batch([
        "When did Bangladesh get independence?",
        "What is the history of Bangladesh?",
        "Who was the first leader of Bangladesh?",
    ])

    end_time = time.time()  # Capture end timestamp
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
