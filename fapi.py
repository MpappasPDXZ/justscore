from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import duckdb
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os

app = FastAPI()

CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "justscorecontainer"

class TeamMetadata(BaseModel):
    team_name: str
    head_coach: str
    age: int
    season: str
    session: str
    created_on: str  # format: 'mm-dd-yyyy'

def get_next_team_number():
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    blobs = container_client.list_blobs(name_starts_with='team_')
    team_numbers = []
    
    for blob in blobs:
        try:
            team_num = int(blob.name.split('/')[0].replace('team_', ''))
            team_numbers.append(team_num)
        except:
            continue
    
    return max(team_numbers + [0]) + 1

@app.post("/create_team/")
async def create_team(team_data: TeamMetadata):
    try:
        # Get next team number
        team_num = get_next_team_number()
        team_folder = f"team_{team_num}"
        
        # Add team_id to the data
        data_dict = team_data.dict()
        data_dict['team_id'] = str(team_num)
        
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Generate filename
        blob_name = f"{team_folder}/project_metadata.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the parquet file
        blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Successfully created team folder and metadata: {blob_name}",
            "team_id": str(team_num),
            "team_data": data_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read_metadata/{team_id}")
async def read_metadata(team_id: str):
    try:
        # Initialize DuckDB
        con = duckdb.connect()
        
        # Install and load required extensions
        con.execute("""
            INSTALL httpfs;
            LOAD httpfs;
            SET enable_http_metadata_cache=true;
            SET enable_object_cache=true;
        """)
        
        # Parse connection string components
        ACCOUNT_NAME = "justscoresa"
        ACCOUNT_KEY = "GUUcJpSTMvebKY0wMPos51Ap2bf6QvRmI8S8FuarKw5TK7JLVgjnLSVZ+NznZP/Bn926jRt6McPp+AStEwtQDQ=="
        
        # Create connection string
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        
        # Register Azure credentials with SSL settings
        con.execute(f"""
            SET azure_storage_connection_string='{connection_string}';
            SET s3_url_style='path';
            SET verify_ssl_certificate=false;
        """)
        
        # Query parquet files using wildcard
        query = f"""
            SELECT *
            FROM read_parquet('azure://{CONTAINER_NAME}/team_{team_id}/*.parquet')
        """
        
        # Execute query and fetch results
        result = con.execute(query).fetchdf()
        
        return {
            "team_id": team_id,
            "metadata": result.to_dict(orient='records')[0]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading metadata for team_{team_id}: {str(e)}"
        )
@app.get("/list_teams")
async def list_teams():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # List all team folders
        teams = set()
        for blob in container_client.list_blobs():
            if blob.name.startswith('team_'):
                team_id = blob.name.split('/')[0]
                teams.add(team_id)
        
        return {"teams": sorted(list(teams))}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing teams: {str(e)}")