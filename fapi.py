from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os

app = FastAPI()

CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "justscorecontainer"

class TeamData(BaseModel):
    team_type: str  # Changed from Literal to str
    team_name: str
    team_affiliation: str
    team_age: int
    team_created_on: str  # format: 'mm-dd-yyyy'
    team_head_coach: str

@app.post("/write_parquet/")
async def write_parquet(team_data: TeamData):
    try:
        # Get next team number
        team_num = get_next_team_number()
        team_folder = f"team_{team_num}"
        
        # Add last_modified to the data
        data_dict = team_data.dict()
        data_dict['team_last_modified'] = datetime.now().strftime('%m-%d-%Y')
        
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload to Azure Blob Storage in team folder
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Generate filename with team folder
        blob_name = f"{team_folder}/data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the parquet file
        blob_client.upload_blob(parquet_buffer.getvalue())
        
        return {
            "message": f"Successfully wrote parquet file: {blob_name}",
            "team_number": team_num,
            "team_data": data_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))