from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

    # TODO: Add validation for the query field
    # For example, you can check if the query is not empty
    