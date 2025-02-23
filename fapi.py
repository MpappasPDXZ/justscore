from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Literal, Optional, List, Dict
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
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
    defensive_position_one: DefensivePosition
    defensive_position_two: Optional[DefensivePosition] = None
    defensive_position_three: Optional[DefensivePosition] = None
    defensive_position_four: Optional[DefensivePosition] = None  # New position
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

class PlayerRank(BaseModel):
    jersey_number: str
    player_rank: int
class DepthChartData(BaseModel):
    team_id: str
    depth_chart: Dict[str, List[PlayerRank]]
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
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Check if team folder exists
        folder_marker = f"{team_folder}/.folder"
        folder_client = container_client.get_blob_client(folder_marker)
        folder_exists = False
        try:
            folder_client.get_blob_properties()
            folder_exists = True
        except Exception:
            # Folder doesn't exist, create it
            folder_client.upload_blob(b"", overwrite=True)
            logger.info(f"Created new team folder: {team_folder}")
        
        # Upload metadata file with team number as name
        metadata_blob_name = f"metadata/{team_num}.parquet"
        metadata_client = container_client.get_blob_client(metadata_blob_name)
        metadata_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": "Successfully updated team metadata" if folder_exists else "Successfully created team folder and metadata",
            "team_id": str(team_num),
            "team_folder": team_folder,
            "metadata_file": metadata_blob_name,
            "team_data": data_dict,
            "folder_status": "existing" if folder_exists else "created"
        }
    except Exception as e:
        logger.error(f"Error in create_team: {str(e)}")
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
@app.put("/teams/{team_id}/metadata")
async def update_team_metadata(team_id: str, team_data: TeamMetadata):
    try:
        # Verify the team exists first
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        metadata_blob_name = f"metadata/{team_id}.parquet"
        metadata_client = container_client.get_blob_client(metadata_blob_name)
        
        try:
            # Check if metadata file exists
            metadata_client.get_blob_properties()
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Team {team_id} not found: {str(e)}"
            )
        
        # Add team_id to the data
        data_dict = team_data.dict()
        data_dict['team_id'] = team_id
        
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload updated metadata
        metadata_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Successfully updated team {team_id} metadata",
            "team_id": team_id,
            "metadata_file": metadata_blob_name,
            "team_data": data_dict
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating team metadata: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating team {team_id} metadata: {str(e)}"
        )
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
        con.execute(f"SET azure_storage_connection_string='{connection_string}';")
        
        # Add union_by_name=True to handle schema differences
        query = f"""
            SELECT *
            FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/*.parquet', union_by_name=True)
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



@app.post("/teams/{team_id}/depth_chart_post")
async def save_depth_chart(team_id: str, depth_chart: DepthChartData):
    try:
        print(f"Saving depth chart for team {team_id}")
        logger.info(f"Saving depth chart for team {team_id}")
        
        # Verify team_id matches
        if team_id != depth_chart.team_id:
            raise HTTPException(
                status_code=400,
                detail="Team ID in path does not match team ID in data"
            )
        
        # Convert to DataFrame format (flattening the nested structure)
        rows = []
        for position, players in depth_chart.depth_chart.items():
            for player in players:
                rows.append({
                    'team_id': team_id,
                    'position': position,
                    'jersey_number': player.jersey_number,
                    'player_rank': player.player_rank
                })
        
        df = pd.DataFrame(rows)
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client("justscorecontainer")
        blob_name = f"teams/team_{team_id}/depth_chart/depth_chart.parquet"
        blob_client = container_client.get_blob_client(blob_name)
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        return {
            "message": "Depth chart saved successfully",
            "team_id": team_id,
            "path": blob_name,
            "positions_saved": list(depth_chart.depth_chart.keys()),
            "total_entries": len(rows)
        }
            
    except Exception as e:
        error_msg = f"Error saving depth chart: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/teams/{team_id}/depth_chart_get")
async def get_depth_chart(team_id: str):
    try:
        # Define position order
        position_order = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        
        # Connect to DuckDB
        con = duckdb.connect()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        con.execute("SET azure_transport_option_type = 'curl';")
        con.execute(f"SET azure_storage_connection_string='{connection_string}';")
        # Get roster data with union_by_name=True
        roster_query = f"""
            SELECT 
                jersey_number,
                player_name,
                active,
                defensive_position_one,
                defensive_position_two,
                defensive_position_three,
                defensive_position_four,
                CAST(defensive_position_allocation_one AS FLOAT) as allocation_one,
                CAST(defensive_position_allocation_two AS FLOAT) as allocation_two,
                CAST(defensive_position_allocation_three AS FLOAT) as allocation_three,
                CAST(defensive_position_allocation_four AS FLOAT) as allocation_four
            FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/*.parquet', union_by_name=True)
            WHERE jersey_number IS NOT NULL
        """
        roster_df = con.execute(roster_query).fetchdf()
        
        # Filter for active players
        active_players = roster_df[roster_df['active'] == 'Active']
        
        try:
            # Try to read existing depth chart
            depth_chart_query = f"""
                SELECT 
                    CAST(team_id AS INTEGER) as team_id,
                    position,
                    player_rank,
                    jersey_number
                FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/depth_chart/depth_chart.parquet')
            """
            depth_chart_df = con.execute(depth_chart_query).fetchdf()
            
            # Remove inactive players from depth chart
            depth_chart_df = depth_chart_df[
                depth_chart_df['jersey_number'].isin(active_players['jersey_number'])
            ]
            
            # Validate existing position assignments
            valid_assignments = []
            for _, row in depth_chart_df.iterrows():
                player = active_players[active_players['jersey_number'] == row['jersey_number']].iloc[0]
                position = row['position']
                
                # Calculate total allocation for this position
                allocation = 0
                if pd.notnull(player['defensive_position_one']) and player['defensive_position_one'] == position:
                    allocation += player['allocation_one'] if pd.notnull(player['allocation_one']) else 0
                if pd.notnull(player['defensive_position_two']) and player['defensive_position_two'] == position:
                    allocation += player['allocation_two'] if pd.notnull(player['allocation_two']) else 0
                if pd.notnull(player['defensive_position_three']) and player['defensive_position_three'] == position:
                    allocation += player['allocation_three'] if pd.notnull(player['allocation_three']) else 0
                if pd.notnull(player['defensive_position_four']) and player['defensive_position_four'] == position:
                    allocation += player['allocation_four'] if pd.notnull(player['allocation_four']) else 0
                
                # Only keep assignments where player has allocation
                if allocation > 0:
                    valid_assignments.append(row)
            
            # Update depth chart with only valid assignments
            depth_chart_df = pd.DataFrame(valid_assignments)
            
            # Check all active players for their positions
            position_assignments = []
            for _, player in active_players.iterrows():
                # Check each position and allocation
                positions_to_check = [
                    ('defensive_position_one', 'allocation_one'),
                    ('defensive_position_two', 'allocation_two'),
                    ('defensive_position_three', 'allocation_three'),
                    ('defensive_position_four', 'allocation_four')
                ]
                
                for pos_field, alloc_field in positions_to_check:
                    if pd.notnull(player[pos_field]) and pd.notnull(player[alloc_field]) and player[alloc_field] > 0:
                        position = player[pos_field]
                        # Check if player is already in this position in depth chart
                        player_in_position = depth_chart_df[
                            (depth_chart_df['jersey_number'] == player['jersey_number']) & 
                            (depth_chart_df['position'] == position)
                        ]
                        
                        if player_in_position.empty:
                            # Player should be in this position but isn't
                            existing_players = depth_chart_df[depth_chart_df['position'] == position]
                            max_rank = existing_players['player_rank'].max() if not existing_players.empty else 0
                            
                            position_assignments.append({
                                'team_id': int(team_id),
                                'position': position,
                                'player_rank': max_rank + 1,
                                'jersey_number': player['jersey_number']
                            })
            
            # Add any missing position assignments
            if position_assignments:
                logger.info(f"Adding {len(position_assignments)} missing position assignments")
                new_assignments_df = pd.DataFrame(position_assignments)
                depth_chart_df = pd.concat([depth_chart_df, new_assignments_df], ignore_index=True)
            
        except Exception as e:
            logger.info(f"No existing depth chart found for team {team_id}, creating default")
            # Create default depth chart based on allocations
            depth_chart_rows = []
            
            # Calculate weighted sum for tiebreaking
            active_players['weighted_sum'] = (
                active_players['allocation_one'].fillna(0) * 1.1 +
                active_players['allocation_two'].fillna(0) * 1.05 +
                active_players['allocation_three'].fillna(0) * 1.03 +
                active_players['allocation_four'].fillna(0)
            )
            
            # Process each position
            for pos in position_order:
                # Get players who can play this position
                pos_players = []
                for _, player in active_players.iterrows():
                    allocation = 0
                    # Check each position with null handling
                    if pd.notnull(player['defensive_position_one']) and player['defensive_position_one'] == pos:
                        allocation += player['allocation_one'] if pd.notnull(player['allocation_one']) else 0
                    if pd.notnull(player['defensive_position_two']) and player['defensive_position_two'] == pos:
                        allocation += player['allocation_two'] if pd.notnull(player['allocation_two']) else 0
                    if pd.notnull(player['defensive_position_three']) and player['defensive_position_three'] == pos:
                        allocation += player['allocation_three'] if pd.notnull(player['allocation_three']) else 0
                    if pd.notnull(player['defensive_position_four']) and player['defensive_position_four'] == pos:
                        allocation += player['allocation_four'] if pd.notnull(player['allocation_four']) else 0
                    
                    if allocation > 0:
                        pos_players.append({
                            'jersey_number': player['jersey_number'],
                            'allocation': allocation,
                            'weighted_sum': player['weighted_sum']
                        })
                
                # Sort players by allocation and weighted sum
                pos_players.sort(key=lambda x: (-x['allocation'], -x['weighted_sum'], x['jersey_number']))
                
                # Add to depth chart
                for rank, player in enumerate(pos_players, 1):
                    depth_chart_rows.append({
                        'team_id': int(team_id),
                        'position': pos,
                        'player_rank': rank,
                        'jersey_number': player['jersey_number']
                    })
            
            depth_chart_df = pd.DataFrame(depth_chart_rows)
        
        # Merge with player names
        result_df = pd.merge(
            depth_chart_df,
            active_players[['jersey_number', 'player_name']],
            on='jersey_number',
            how='left'
        )
        
        # Create player display name
        result_df['player_name'] = result_df.apply(
            lambda x: f"{x['jersey_number']} - {x['player_name']}", axis=1
        )
        
        # Sort by position order and rank
        result_df['position_order'] = result_df['position'].map({pos: idx for idx, pos in enumerate(position_order)})
        result_df = result_df.sort_values(['position_order', 'player_rank'])
        result_df = result_df.drop('position_order', axis=1)
        
        # Convert to dictionary format
        result = result_df.to_dict(orient='records')
        
        return {
            "team_id": int(team_id),
            "depth_chart": result
        }
            
    except Exception as e:
        logger.error(f"Error in get_depth_chart: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading depth chart for team {team_id}: {str(e)}"
        )

@app.delete("/teams/{team_id}")
async def delete_team(team_id: str):
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        deleted_files = []
        errors = []

        # 1. Delete metadata file
        try:
            metadata_blob_name = f"metadata/{team_id}.parquet"
            metadata_blob_client = container_client.get_blob_client(metadata_blob_name)
            metadata_blob_client.delete_blob()
            deleted_files.append(metadata_blob_name)
        except Exception as e:
            errors.append(f"Error deleting metadata: {str(e)}")

        # 2. Delete team folder and all its contents
        try:
            # List all blobs in the team folder
            team_folder_prefix = f"teams/team_{team_id}/"
            blobs_list = container_client.list_blobs(name_starts_with=team_folder_prefix)
            
            # Delete each blob in the folder
            for blob in blobs_list:
                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()
                    deleted_files.append(blob.name)
                except Exception as e:
                    errors.append(f"Error deleting {blob.name}: {str(e)}")
        except Exception as e:
            errors.append(f"Error accessing team folder: {str(e)}")

        # Check if any files were deleted
        if not deleted_files:
            raise HTTPException(
                status_code=404,
                detail=f"Team {team_id} not found or already deleted"
            )

        # Return status with details
        response = {
            "message": f"Team {team_id} deleted successfully",
            "team_id": team_id,
            "deleted_files": deleted_files,
        }
        
        # Include errors in response if any occurred
        if errors:
            response["warnings"] = errors

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting team {team_id}: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting team {team_id}: {str(e)}"
        )
@app.get("/hello")
async def hello():
    print("Hello endpoint called!  Version 2")
    return {"message": "Hello World!  Version 2"}