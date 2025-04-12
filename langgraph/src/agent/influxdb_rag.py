"""A simple rag that query from InfluxDB and display the results in natural language.

At a high-level, the steps of these systems are:
    1. Convert question to SQL query: Model converts user input to a SQL query.
    2. Execute SQL query: Execute the query.
    3. Answer the question: Model responds to user input using the query results.
"""

from typing import cast
from helper.influxdb import InfluxDB
from state.influxdb_qa import State
from training.influxdb_few_shot import InfluxDBFewShot
from influxdb_client_3 import InfluxDBClient3
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from helper.prompt import Prompt
from model.influxdb_output import QueryOutput, AnswerOutput
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()  # loading .env

llm = ChatOllama(model="llama3.2", temperature=0.01)  # Initialize the model

# Initialize the client
client = InfluxDBClient3(
    token=os.getenv("INFLUXDB_TOKEN"),
    host=os.getenv("INFLUXDB_HOST"),
    org=os.getenv("INFLUXDB_ORG"),
    database=os.getenv("INFLUXDB_DB"),
)

# Get the tables, columns, and data from InfluxDB
tables = InfluxDB.tables(client)
columns = InfluxDB.columns(client, tables)
data = InfluxDB.data(client, tables)


def querify(state: State) -> State:
    """Generate SQL query from user input question."""
    prompt_template = ChatPromptTemplate(
        [
            ("system", "{instruction}"),
            InfluxDBFewShot.sql_generation_train(),
            ("human", "Question: {question}"),
        ],
    )

    prompt = prompt_template.with_config(run_name="Prompt").invoke(
        {
            "instruction": Prompt.prompt_sql_query_generation().format(
                top_k=10,
                table_info=tables,
                column_info=columns,
                sample_data=data,
                error_list="",
            ),
            "question": state["question"],
        },
    )

    try:
        response = (
            llm.with_structured_output(schema=QueryOutput)
            .with_config(run_name="Querify")
            .invoke(prompt)
        )

        return cast(State, response.model_dump())
    except Exception as e:
        raise RuntimeError("Error:", e) from e


@traceable(name="Execute SQL", run_type="tool")
def execute(state: State) -> State:
    """Execute SQL query."""
    return {"result": client.query(state["query"]).to_pylist()}


def answer_up(state: State) -> State:
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )

    try:
        response = (
            llm.with_structured_output(schema=AnswerOutput)
            .with_config(run_name="Answer up")
            .invoke(prompt)
        )

        return cast(State, response.model_dump())
    except Exception as e:
        raise RuntimeError("Error:", e) from e


def dag() -> CompiledStateGraph:
    """Create and compile the state graph."""
    workflow = StateGraph(State)
    workflow.add_node("querify", querify)
    workflow.add_node("execute", execute)
    workflow.add_node("answer_up", answer_up)
    workflow.add_edge(START, "querify")
    workflow.add_edge("querify", "execute")
    workflow.add_edge("execute", "answer_up")
    workflow.add_edge("answer_up", END)

    return workflow.compile(name="InfluxDB RAG")


try:
    influxdb_rag = dag()
except Exception as e:
    raise RuntimeError("Error: ", e) from e
