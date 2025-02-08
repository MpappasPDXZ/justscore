from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI()

# Basic GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World from Git1"}

# Path parameter example
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# Query parameter example
@app.get("/search/")
def search_items(query: str = None, skip: int = 0, limit: int = 10):
    return {
        "query": query,
        "skip": skip,
        "limit": limit
    }

# Define a data model
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

# POST endpoint with request body
@app.post("/items/")
def create_item(item: Item):
    return item

# To run this:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)