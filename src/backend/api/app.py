from fastapi import FastAPI, HTTPException
from datetime import datetime

from src.backend.schema.api_data_model import QueryRequest
from src.backend.rag import RAG_base
# Initialize FastAPI app
app = FastAPI()




@app.post("/generate_response")
def generate_response(request: QueryRequest):
    query = request.query
    response = generate_ai_response(query)
    timestamp = datetime.now().isoformat()

    try:
       pass
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to log interaction")

    return {"query": query, "response": response, "timestamp": timestamp}
