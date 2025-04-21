from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Database setup
DB_PATH = "responses.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Request model
class QueryRequest(BaseModel):
    query: str

# Mock AI response generator
def generate_ai_response(query: str) -> str:
    return f"AI response to: {query}"


@app.post("/generate_response")
def generate_response(request: QueryRequest):
    query = request.query
    response = generate_ai_response(query)
    timestamp = datetime.now().isoformat()

    # Log interaction in the database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (query, response, timestamp)
            VALUES (?, ?, ?)
        """, (query, response, timestamp))
        conn.commit()
        conn.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to log interaction")

    return {"query": query, "response": response, "timestamp": timestamp}
