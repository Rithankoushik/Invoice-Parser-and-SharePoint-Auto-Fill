"""
Database Loader Module
Handles structured data retrieval using SQL
"""

from sqlalchemy import create_engine, text
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


def create_sql_retriever(db_path: str, table_name: str = None):
    """
    Create a SQL agent for natural language queries to database

    Args:
        db_path: Path to SQLite database (e.g., "data/mydatabase.db")
        table_name: Optional table name for direct queries

    Returns:
        SQL agent that can answer questions about the database
    """
    # Create database connection
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)

    # Use Groq LLM for SQL agent
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Create SQL agent
    agent = create_sql_agent(
        llm=llm,
        engine=engine,
        verbose=True
    )

    return agent


def get_table_info(db_path: str):
    """Get information about tables in the database"""
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        # Get all table names
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        tables = [row[0] for row in result]

        # Get schema for each table
        schema_info = {}
        for table in tables:
            result = conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [{"name": row[1], "type": row[2]} for row in result]
            schema_info[table] = columns

        return {"tables": tables, "schema": schema_info}


def execute_raw_query(db_path: str, query: str):
    """Execute a raw SQL query and return results"""
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        result = conn.execute(text(query))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result]

        return {"columns": columns, "rows": rows}
