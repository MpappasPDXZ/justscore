from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
from typing import Optional
import duckdb
from dotenv import load_dotenv
load_dotenv()

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
@app.get("/max_team_number")
async def get_max_team_number():
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.list_blobs(name_starts_with='teams/team_')
        team_numbers = []
        for blob in blobs:
            try:
                # Extract number from folder name (e.g., "teams/team_3/something" -> 3)
                folder_name = blob.name.split('/')[1]  # gets "team_3"
                team_num = int(folder_name.split('_')[1])  # gets "3"
                team_numbers.append(team_num)
            except:
                continue
        if not team_numbers:
            return {
                "max_team_number": 0,
                "next_team_number": 1,
                "message": "No teams found"
            }
        max_num = max(team_numbers)
        return {
            "max_team_number": max_num,
            "next_team_number": max_num + 1,
            "message": f"Highest team number is {max_num}, next available is {max_num + 1}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding max team number: {str(e)}"
        )

@app.post("/create_team/")
async def create_team(team_data: TeamMetadata):
    try:
        # Get next team number
        team_num = get_max_team_number()
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
@app.get("/read_metadata/{team_id}")
async def read_metadata(team_id: str):
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_name = f"metadata/{team_id}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
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
    
@app.get("/read_metadata_duckdb")
async def read_metadata_duckdb():
    try:
        con = duckdb.connect()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        con.execute("SET azure_transport_option_type = 'curl';")
        con.execute(f"""
            SET azure_storage_connection_string='{connection_string}';
        """)
        query = f"""
            SELECT *
            FROM read_parquet('azure://{CONTAINER_NAME}/metadata/*.parquet')
            ORDER BY CAST(team_id AS INTEGER)
        """
        result = con.execute(query).fetchdf()
        if result.empty:
            return {
                "message": "No teams found",
                "metadata": []
            }
        
        return {
            "total_teams": len(result),
            "metadata": result.to_dict(orient='records')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading metadata: {str(e)}"
        )
    
@app.get("/teams/{team_id}/roster")
async def get_team_roster(team_id: str):
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_name = f"teams/team_{team_id}/roster.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        blob_data  = blob_client.download_blob().readall()
        parquet_file = BytesIO(blob_data)
        df = pd.read_parquet(parquet_file)
        float_columns = [
            'defensive_position_allocation_one',
            'defensive_position_allocation_two',
            'defensive_position_allocation_three'
        ]
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
        
        roster_dict = df.to_dict(orient='records')
        return {
            "team_id": team_id,
            "roster": roster_dict  # Use roster_dict instead of df.to_dict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Team {team_id}/roster.parquet not found or error reading metadata: {str(e)}"
        )