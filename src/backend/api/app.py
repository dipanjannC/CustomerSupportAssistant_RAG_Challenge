from fastapi import FastAPI, HTTPException
from datetime import datetime
import uvicorn

from src.backend.schema.api_data_model import QueryRequest
from src.backend.vectorstore import get_vectorstore_instance
from src.backend.rag import RAG_base

app = FastAPI()

vectorstore = get_vectorstore_instance()


@app.post("/generate_response")
async def generate_response(request: QueryRequest) -> dict:
    """
    Endpoint to generate a response based on the input query.
    It is using the Retrieval-Augmented Generation (RAG) pipeline.
    The function retrieves relevant documents from a vectorstore based on the input query,
    and then generates a response using a llm with the retrieved context.

    Args:
        request (QueryRequest): The input query request containing the query text.

    Returns:
        dict: A dictionary containing the query, response, and timestamp.

    Raises:
        HTTPException: If there is an error during processing.
    """
   
    
    timestamp = datetime.now().isoformat()

    try:
      
      response = RAG_base(query_text=request.query,retriever=vectorstore)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"query": request.query, "response": response, "timestamp": timestamp}


if __name__ == "__main__":

    uvicorn.run("src.backend.api.app:app", host="0.0.0.0", port=8085,reload=True)