from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import json

app = FastAPI()

# Your connection string from Azure Storage Account
CONNECTION_STRING = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=justscoresa;AccountKey=GUUcJpSTMvebKY0wMPos51Ap2bf6QvRmI8S8FuarKw5TK7JLVgjnLSVZ+NznZP/Bn926jRt6McPp+AStEwtQDQ==;BlobEndpoint=https://justscoresa.blob.core.windows.net/;FileEndpoint=https://justscoresa.file.core.windows.net/;QueueEndpoint=https://justscoresa.queue.core.windows.net/;TableEndpoint=https://justscoresa.table.core.windows.net/"
CONTAINER_NAME = "justscorecontainer"

class JsonData(BaseModel):
    data: dict

@app.post("/write_parquet/")
async def write_parquet(json_data: JsonData):
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame([json_data.data])
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Generate unique filename
        blob_name = f"data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the parquet file
        blob_client.upload_blob(parquet_buffer.getvalue())
        
        return {"message": f"Successfully wrote parquet file: {blob_name}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep your existing endpoints
@app.get("/")
def read_root():
    return {"message": "Hello World from Matt"}