from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
from typing import Optional

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI Team Management API"}

class TeamMetadata(BaseModel):
    team_name: str
    head_coach: str
    age: int
    season: str
    session: str
    created_on: str  # format: 'mm-dd-yyyy'

class Player(BaseModel):
    player_name: str
    jersey_number: str
    active: str
    defensive_position_one: Optional[str] = None
    defensive_position_two: Optional[str] = None
    defensive_position_three: Optional[str] = None
    defensive_position_allocation_one: Optional[float] = None
    defensive_position_allocation_two: Optional[float] = None
    defensive_position_allocation_three: Optional[float] = None

class TeamRoster(BaseModel):
    players: list[Player]

# Get Azure credentials from environment variables
ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
ACCOUNT_KEY = os.getenv('AZURE_STORAGE_KEY')
CONTAINER_NAME = "justscorecontainer"

def get_blob_service_client():
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def get_next_team_number():
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # Look in metadata folder for highest team number
    blobs = container_client.list_blobs(name_starts_with='metadata/')
    team_numbers = []
    
    for blob in blobs:
        try:
            # Extract number from filename (e.g., "metadata/3.parquet" -> 3)
            team_num = int(blob.name.split('/')[-1].replace('.parquet', ''))
            team_numbers.append(team_num)
        except:
            continue
    
    return max(team_numbers + [0]) + 1

@app.post("/create_team/")
async def create_team(team_data: TeamMetadata):
    try:
        # Get next team number
        team_num = get_next_team_number()
        team_folder = f"teams/team_{team_num}"
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
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Create team folder (empty blob as folder marker)
        folder_marker = f"{team_folder}/.folder"
        folder_client = container_client.get_blob_client(folder_marker)
        folder_client.upload_blob(b"", overwrite=True)
        # Upload metadata file with team number as name
        metadata_blob_name = f"metadata/{team_num}.parquet"
        metadata_client = container_client.get_blob_client(metadata_blob_name)
        metadata_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        return {
            "message": f"Successfully created team folder and metadata",
            "team_id": str(team_num),
            "team_folder": team_folder,
            "metadata_file": metadata_blob_name,
            "team_data": data_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/read_metadata/{team_id}")
async def read_metadata(team_id: str):
    try:
        # Create blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Get blob client for the metadata file
        blob_name = f"metadata/{team_id}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        # Download the blob
        blob_data = blob_client.download_blob().readall()
        # Read parquet from memory
        parquet_file = BytesIO(blob_data)
        df = pd.read_parquet(parquet_file)
        return {
            "team_id": team_id,
            "metadata": df.to_dict(orient='records')[0]
        }
    except Exception as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Team {team_id} not found or error reading metadata: {str(e)}"
        )
    
@app.post("/teams/{team_id}/roster")
async def create_team_roster(team_id: str, roster: TeamRoster):
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Convert roster to DataFrame and add team_id
        players_data = [player.dict() for player in roster.players]
        df = pd.DataFrame(players_data)
        df['team_id'] = team_id  # Add team_id column
        # Ensure all columns exist with correct order
        columns = [
            'team_id',
            'player_name',
            'jersey_number',
            'active',
            'defensive_position_one',
            'defensive_position_two',
            'defensive_position_three',
            'defensive_position_allocation_one',
            'defensive_position_allocation_two',
            'defensive_position_allocation_three'
        ]
        # Reorder columns and fill missing values with None
        df = df.reindex(columns=columns)
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        # Upload roster file to team folder
        blob_name = f"teams/team_{team_id}/roster.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        return {
            "message": f"Successfully created roster for team_{team_id}",
            "team_id": team_id,
            "player_count": len(roster.players)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating roster: {str(e)}"
        )
    
@app.get("/teams/{team_id}/roster")
async def get_team_roster(team_id: str):
    try:
        # Check if environment variables are set
        if not ACCOUNT_NAME or not ACCOUNT_KEY:
            raise HTTPException(
                status_code=500,
                detail="Azure storage credentials not properly configured"
            )
        try:
            blob_service_client = get_blob_service_client()
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to Azure storage: {str(e)}"
            )
        # Get roster file from team folder
        blob_name = f"teams/team_{team_id}/roster.parquet"
        try:
            # Check if blob exists
            blob_client = container_client.get_blob_client(blob_name)
            if not blob_client.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"No roster found for team_{team_id}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error checking if roster exists: {str(e)}"
            )
        try:
            # Download and read the roster
            blob_data = blob_client.download_blob().readall()
            parquet_file = BytesIO(blob_data)
            df = pd.read_parquet(parquet_file)
            return {
                "team_id": team_id,
                "roster": df.to_dict(orient='records')
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading roster parquet file: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
