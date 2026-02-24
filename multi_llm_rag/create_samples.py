"""
Create sample database and documents for testing
"""

import sqlite3
import os


def create_sample_database(db_path="data/mydatabase.db"):
    """Create a sample SQLite database with test data"""

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            department TEXT,
            salary INTEGER,
            join_date TEXT
        )
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER,
            created_at TEXT
        )
    """)

    # Create sales table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            user_id INTEGER,
            quantity INTEGER,
            total_amount REAL,
            sale_date TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Insert sample data - Users
    users_data = [
        ("John Doe", "john@example.com", 28, "Engineering", 75000, "2023-01-15"),
        ("Jane Smith", "jane@example.com", 32, "Marketing", 65000, "2022-06-20"),
        ("Bob Johnson", "bob@example.com", 45, "Sales", 80000, "2020-03-10"),
        ("Alice Williams", "alice@example.com", 29, "Engineering", 70000, "2023-08-01"),
        ("Charlie Brown", "charlie@example.com", 35, "HR", 55000, "2021-11-25"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO users (name, email, age, department, salary, join_date) VALUES (?, ?, ?, ?, ?, ?)",
        users_data
    )

    # Insert sample data - Products
    products_data = [
        ("Laptop Pro", "Electronics", 1299.99, 50, "2024-01-10"),
        ("Wireless Mouse", "Accessories", 29.99, 200, "2024-02-15"),
        ("Monitor 27\"", "Electronics", 399.99, 75, "2024-01-20"),
        ("Keyboard Mechanical", "Accessories", 149.99, 120, "2024-03-01"),
        ("USB-C Hub", "Accessories", 49.99, 180, "2024-02-28"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO products (name, category, price, stock, created_at) VALUES (?, ?, ?, ?, ?)",
        products_data
    )

    # Insert sample data - Sales
    sales_data = [
        (1, 1, 2, 2599.98, "2024-06-01"),
        (2, 1, 3, 89.97, "2024-06-02"),
        (3, 2, 1, 399.99, "2024-06-03"),
        (4, 3, 1, 149.99, "2024-06-04"),
        (5, 4, 2, 1099.98, "2024-06-05"),
        (1, 3, 1, 1299.99, "2024-06-06"),
        (2, 5, 5, 149.95, "2024-06-07"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO sales (product_id, user_id, quantity, total_amount, sale_date) VALUES (?, ?, ?, ?, ?)",
        sales_data
    )

    conn.commit()
    conn.close()

    print(f"Database created: {db_path}")
    print("Tables: users, products, sales")


def create_sample_document(doc_path="data/document.txt"):
    """Create a sample text document"""
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    content = """
Multi-LLM RAG System Documentation

Introduction
============
This is a Retrieval-Augmented Generation (RAG) system that uses multiple
language models to provide accurate answers based on your documents and data.

Key Features
============
1. Multi-Model Architecture
   - Uses different Groq models for different tasks
   - Llama 3.1 8B for fast responses
   - Mixtral 8x7B for reasoning tasks
   - Llama 3.1 70B for complex code generation

2. Hybrid Retrieval
   - Document retrieval from PDF, TXT, DOCX files
   - SQL database querying for structured data
   - Automatic routing between retrieval methods

3. Vector Storage
   - ChromaDB for efficient vector storage
   - HuggingFace embeddings for semantic search
   - Fast similarity search

How It Works
============
1. User submits a query
2. System classifies the task type (reasoning/code/fast)
3. Relevant context is retrieved from documents and/or database
4. Appropriate LLM generates answer based on context

Use Cases
=========
- Customer support with company documentation
- Internal knowledge base Q&A
- Code assistant for programming questions
- Data analysis with natural language

Getting Started
==============
1. Add your Groq API key to .env file
2. Place documents in the data/ folder
3. Optionally create a SQLite database
4. Run the application

For more information, visit the documentation.
"""

    with open(doc_path, 'w') as f:
        f.write(content)

    print(f"Document created: {doc_path}")


if __name__ == "__main__":
    create_sample_database()
    create_sample_document()
    print("\nSample data created successfully!")
