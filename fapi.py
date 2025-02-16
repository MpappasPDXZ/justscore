from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Literal, Optional
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
from typing import Optional
import duckdb
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)
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
# Define valid defensive positions
DefensivePosition = Literal[
    'P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH','DP', 'FL'
]

class PlayerData(BaseModel):
    player_name: str
    jersey_number: str
    active: str  # "Active" or "Inactive"
    defensive_position_one: str
    defensive_position_two: Optional[str] = None
    defensive_position_three: Optional[str] = None
    defensive_position_four: Optional[str] = None  # New position
    defensive_position_allocation_one: str
    defensive_position_allocation_two: Optional[str] = None
    defensive_position_allocation_three: Optional[str] = None
    defensive_position_allocation_four: Optional[str] = None  # New allocation

    @field_validator('active')
    @classmethod
    def validate_active(cls, v):
        if v not in ["Active", "Inactive"]:
            raise ValueError('active must be either "Active" or "Inactive"')
        return v

    @field_validator('defensive_position_allocation_one', 'defensive_position_allocation_two', 
                    'defensive_position_allocation_three', 'defensive_position_allocation_four')
    @classmethod
    def validate_allocations(cls, v):
        if v is not None:
            try:
                float_val = float(v)
                if float_val < 0 or float_val > 1:
                    raise ValueError('Allocation must be between 0 and 1')
                return f"{float_val:.2f}"
            except ValueError:
                raise ValueError('Allocation must be a valid number between 0 and 1')
        return v

class TeamRoster(BaseModel):
    players: list[PlayerData]

# Get Azure credentials from environment variables
ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
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
        max_team_info = await get_max_team_number()
        team_num = max_team_info["next_team_number"]
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
# Add an OPTIONS endpoint handler
@app.options("/read_metadata_duckdb")
async def options_metadata():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://justscorereact.delightfulsky-cfea119e.centralus.azurecontainerapps.io",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@app.delete("/teams/{team_id}/player/{jersey_number}")
async def delete_player(team_id: str, jersey_number: str):
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_name = f"teams/team_{team_id}/{jersey_number}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        try:
            blob_client.get_blob_properties()
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Player with jersey number {jersey_number} not found in team {team_id}"
            )
        blob_client.delete_blob()
        
        return {
            "message": f"Successfully deleted player with jersey number {jersey_number} from team {team_id}",
            "team_id": team_id,
            "jersey_number": jersey_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting player from team {team_id}: {str(e)}"
        )
@app.post("/teams/{team_id}/player")
async def add_or_edit_player(team_id: str, player: PlayerData):
    try:
        # Convert player to dict and add IDs
        player_dict = player.model_dump()
        player_dict['team_id'] = team_id
        player_dict['player_id'] = f"{team_id}_{player.jersey_number}"
        player_dict['last_modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if player already exists
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_name = f"teams/team_{team_id}/{player.jersey_number}.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        
        action_message = "added"
        try:
            existing_blob = blob_client.download_blob().readall()
            existing_df = pd.read_parquet(BytesIO(existing_blob))
            action_message = "updated"
            # Preserve created_on from existing data
            player_dict['created_on'] = existing_df['created_on'].iloc[0]
        except Exception:
            player_dict['created_on'] = player_dict['last_modified']
        
        # Convert to DataFrame
        df = pd.DataFrame([player_dict])
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the parquet file
        blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Player {action_message} successfully",
            "team_id": team_id,
            "player": player_dict,
            "action": action_message
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding/updating player in team {team_id}: {str(e)}"
        )
@app.get("/teams/{team_id}/roster")
async def get_team_roster(team_id: str):
    try:
        con = duckdb.connect()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        con.execute("SET azure_transport_option_type = 'curl';")
        con.execute(f"""
            SET azure_storage_connection_string='{connection_string}';
        """)
        query = f"""
            SELECT *
            FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/*.parquet')
        """
        result = con.execute(query).fetchdf()
        if result.empty:
            return {
                "team_id": team_id,
                "message": "No roster found",
                "roster": []
            }
        
        # Process allocations and positions
        allocation_columns = [
            'defensive_position_allocation_one',
            'defensive_position_allocation_two',
            'defensive_position_allocation_three',
            'defensive_position_allocation_four'
        ]
        
        for col in allocation_columns:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: f"{float(x):.2f}" if pd.notnull(x) else None
                )
        
        position_columns = [
            'defensive_position_one',
            'defensive_position_two',
            'defensive_position_three',
            'defensive_position_four'
        ]
        
        for col in position_columns:
            if col in result.columns:
                result[col] = result[col].astype(str).where(pd.notnull(result[col]), None)
        
        if 'team_id' in result.columns:
            result['team_id'] = result['team_id'].astype(str)
        
        if 'jersey_number' in result.columns:
            result['jersey_number'] = result['jersey_number'].astype(str)
        
        return {
            "team_id": team_id,
            "roster": result.to_dict(orient='records')
        }
            
    except Exception as e:
        logger.error(f"Error in get_team_roster: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading team {team_id} roster: {str(e)}"
        )

@app.get("/test/roster")
async def test_hardcoded_roster():
    try:
        print("Starting test roster request for hardcoded team 1")  # Console log
        logger.info("Starting test roster request for hardcoded team 1")  # File log
        
        con = duckdb.connect()
        print("DuckDB connected")
        logger.info("DuckDB connected")
        
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        print("Connection string created")
        logger.info("Connection string created")
        
        con.execute("SET azure_transport_option_type = 'curl';")
        print("Transport type set")
        logger.info("Transport type set")
        
        con.execute(f"""
            SET azure_storage_connection_string='{connection_string}';
        """)
        print("Storage connection set")
        logger.info("Storage connection set")
        
        query = """
            SELECT *
            FROM read_parquet('azure://justscorecontainer/teams/team_1/[0-9]*.parquet', union_by_name=True)
        """
            
        print(f"Query to execute: {query}")
        logger.info(f"Query to execute: {query}")
        
        result = con.execute(query).fetchdf()
        print(f"Query executed successfully. Found {len(result)} rows")
        logger.info(f"Query executed successfully. Found {len(result)} rows")
        
        if result.empty:
            return {
                "team_id": "1",
                "message": "No roster found",
                "roster": []
            }
        
        # Process the data
        allocation_columns = [
            'defensive_position_allocation_one',
            'defensive_position_allocation_two',
            'defensive_position_allocation_three',
            'defensive_position_allocation_four'
        ]
        
        for col in allocation_columns:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: f"{float(x):.2f}" if pd.notnull(x) else None
                )
        
        position_columns = [
            'defensive_position_one',
            'defensive_position_two',
            'defensive_position_three',
            'defensive_position_four'
        ]
        
        for col in position_columns:
            if col in result.columns:
                result[col] = result[col].astype(str).where(pd.notnull(result[col]), None)
        
        if 'team_id' in result.columns:
            result['team_id'] = result['team_id'].astype(str)
        
        if 'jersey_number' in result.columns:
            result['jersey_number'] = result['jersey_number'].astype(str)
        
        return {
            "team_id": "1",
            "roster": result.to_dict(orient='records')
        }
            
    except Exception as e:
        error_msg = f"Error in test_hardcoded_roster: {str(e)}"
        print(error_msg)  # Console log
        logger.error(error_msg)  # File log
        print(f"Error type: {type(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        tb = traceback.format_exc()
        print(f"Traceback: {tb}")
        logger.error(f"Traceback: {tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading hardcoded team 1 roster: {str(e)}"
        )
@app.get("/hello")
async def hello():
    print("Hello endpoint called!")
    logger.info("Hello endpoint called!")
    return {"message": "Hello World!"}