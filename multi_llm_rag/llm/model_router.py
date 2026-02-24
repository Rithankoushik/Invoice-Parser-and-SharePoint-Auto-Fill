"""
Model Router Module
Routes queries to appropriate Groq models based on task type
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

load_dotenv()


class ModelRouter:
    """
    Routes queries to different Groq models based on task type:
    - reasoning: Complex analysis, explanations
    - fast: Quick simple responses
    - code: Programming tasks
    """

    def __init__(self):
        # Initialize Groq models
        # Model 1: Llama 3.1 8B - Fast and efficient for general tasks
        self.fast_model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Model 2: Mixtral 8x7B - Good for reasoning and analysis
        self.reasoning_model = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Model 3: Llama 3.1 70B - Best for complex code tasks
        self.code_model = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def decide_task_type(self, query: str) -> str:
        """
        Classify the query into task type

        Returns:
            'reasoning' for analysis/explanation queries
            'code' for programming tasks
            'fast' for simple queries
        """
        query_lower = query.lower()

        # Keywords for code tasks
        code_keywords = [
            "code", "function", "implement", "write", "program",
            "python", "javascript", "java", "class", "algorithm",
            "debug", "fix", "create", "build"
        ]

        # Keywords for reasoning tasks
        reasoning_keywords = [
            "explain", "why", "how", "analyze", "compare",
            "difference between", "describe", "elaborate", "reason"
        ]

        # Check for code keywords
        if any(kw in query_lower for kw in code_keywords):
            return "code"

        # Check for reasoning keywords
        if any(kw in query_lower for kw in reasoning_keywords):
            return "reasoning"

        # Default to fast model
        return "fast"

    def route(self, query: str, task_type: str = None) -> str:
        """
        Route query to appropriate model based on task type

        Args:
            query: The user's question
            task_type: Optional override for task type

        Returns:
            Model's response as string
        """
        if task_type is None:
            task_type = self.decide_task_type(query)

        print(f"Routing to {task_type} model...")

        if task_type == "reasoning":
            response = self.reasoning_model.invoke(query)
        elif task_type == "code":
            response = self.code_model.invoke(query)
        else:  # fast
            response = self.fast_model.invoke(query)

        return response.content

    def route_with_context(self, query: str, context: str, task_type: str = None) -> str:
        """
        Route query with additional context (for RAG)

        Args:
            query: The user's question
            context: Retrieved context from documents/database
            task_type: Optional task type override

        Returns:
            Model's response
        """
        if task_type is None:
            task_type = self.decide_task_type(query)

        # Build prompt with context
        prompt = f"""Based on the following context, answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""

        return self.route(prompt, task_type)


# Example usage
if __name__ == "__main__":
    router = ModelRouter()

    # Test task classification
    test_queries = [
        "Explain how photosynthesis works",
        "Write a Python function to reverse a string",
        "What is the weather today?"
    ]

    for q in test_queries:
        task = router.decide_task_type(q)
        print(f"Query: {q}")
        print(f"Task type: {task}\n")
