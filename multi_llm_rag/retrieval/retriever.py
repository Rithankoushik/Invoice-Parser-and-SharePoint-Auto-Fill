"""
Retriever Module
Handles hybrid retrieval from documents and database
"""

import os
from typing import Dict, List, Any


class HybridRetriever:
    """
    Combines vector store retrieval with SQL database queries
    """

    def __init__(self, vectorstore, sql_agent=None):
        """
        Initialize hybrid retriever

        Args:
            vectorstore: Chroma vector store for document retrieval
            sql_agent: Optional SQL agent for database queries
        """
        self.vectorstore = vectorstore
        self.sql_agent = sql_agent
        self.document_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def retrieve_documents(self, query: str) -> List:
        """Retrieve relevant documents from vector store"""
        return self.document_retriever.invoke(query)

    def retrieve_sql(self, query: str) -> str:
        """Execute natural language query against SQL database"""
        if self.sql_agent is None:
            return "SQL agent not configured"

        try:
            response = self.sql_agent.run(query)
            return response
        except Exception as e:
            return f"SQL query failed: {str(e)}"

    def hybrid_search(self, query: str, use_sql: bool = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining document and SQL retrieval

        Args:
            query: User's question
            use_sql: Override SQL usage (auto-detect if None)

        Returns:
            Dictionary with documents and SQL results
        """
        # Auto-detect if SQL is needed
        if use_sql is None:
            use_sql = self._needs_sql(query)

        # Retrieve documents
        documents = self.retrieve_documents(query)

        # Retrieve SQL data if needed
        sql_results = ""
        if use_sql and self.sql_agent:
            sql_results = self.retrieve_sql(query)

        return {
            "documents": documents,
            "sql_data": sql_results,
            "used_sql": use_sql
        }

    def _needs_sql(self, query: str) -> bool:
        """Determine if query likely needs SQL data"""
        sql_keywords = [
            "how many", "count", "total", "sum", "average",
            "database", "table", "row", "sql", "query",
            "revenue", "sales", "employees", "users",
            "statistics", "data in", "records"
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in sql_keywords)

    def format_context(self, results: Dict[str, Any]) -> str:
        """
        Format retrieval results into a context string

        Args:
            results: Results from hybrid_search

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add document context
        if results["documents"]:
            doc_texts = "\n\n".join([
                f"[Document {i+1}]: {doc.page_content}"
                for i, doc in enumerate(results["documents"])
            ])
            context_parts.append(f"Document Context:\n{doc_texts}")

        # Add SQL context
        if results["sql_data"] and results["used_sql"]:
            context_parts.append(f"Database Context:\n{results['sql_data']}")

        return "\n\n".join(context_parts) if context_parts else "No relevant context found."
