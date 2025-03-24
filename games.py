from utils import *  # Import all common utilities including the new function
import duckdb
from pydantic import field_validator
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional
import pandas as pd

router = APIRouter()

class GameData(BaseModel):
    user_team: int
    coach: str
    away_team_name: str
    event_date: str
    event_hour: int
    event_minute: int
    field_name: str
    field_location: str
    field_type: str
    field_temperature: str
    game_status: str
    my_team_ha: str

    @field_validator('field_temperature')
    @classmethod
    def validate_temperature(cls, v):
        try:
            temp = float(v)
            if temp < -100 or temp > 150:  # Reasonable range for temperatures in °F
                raise ValueError('Temperature must be between -100 and 150')
            return v
        except ValueError:
            raise ValueError('Temperature must be a valid number')

class RosterPlayer(BaseModel):
    jersey_number: str
    name: str
    position: str
    order_number: Optional[int] = None

class GameRoster(BaseModel):
    players: List[RosterPlayer]

@router.get("/{team_id}") 
async def list_games(team_id: str):
    """
    List all games for a specific team by finding game_[number].parquet files
    """
    try:
        con = get_duckdb_connection()
        game_df = pd.DataFrame()  # Initialize an empty DataFrame
        try:
            game_list_blob_name = f"games/team_{team_id}/game_*.parquet"      
            query = f"""
                SELECT 
                *
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_list_blob_name}')
            """
            game_df = con.execute(query).fetchdf()           
            # Handle the case where columns might be missing
            if game_df.empty:
                logger.warning(f"Empty game list for team {team_id}")
                return {
                    "team_id": team_id,
                    "games_count": 0,
                    "games": []
                }
                
        except Exception as e:
            error_msg = str(e)
            if "Failed to open" in error_msg or "No files found" in error_msg:
                logger.info(f"No games found for team {team_id}")
                return {
                    "team_id": team_id,
                    "games_count": 0,
                    "games": []
                }
            else:
                logger.error(f"Error processing game file {team_id}: {str(e)}")
        # Only try to process the DataFrame if it's not empty
        if not game_df.empty:
            return {
                "team_id": team_id,
                "games_count": len(game_df),
                "games": game_df.to_dict(orient='records')
            }
        else:
            return {
                "team_id": team_id,
                "games_count": 0,
                "games": []
            }
    except Exception as e:
        logger.error(f"Error listing games for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing games for team {team_id}: {str(e)}"
        )

@router.get("/{team_id}/{game_id}")
async def get_game(team_id: str, game_id: str):
    """
    Get details for a specific game
    """
    try:
        # Get DuckDB connection
        print("proper api hit")
        con = get_duckdb_connection()
        
        # Get game data with explicit column selection
        game_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        
        # First check if the blob exists
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(game_blob_name)
        
        if not blob_exists(blob_client):
            return {
                "status": "error",
                "message": f"Game {game_id} not found for team {team_id}",
                "detail": "Create a new game with this ID, or choose another game ID."
            }
            
        try:
            # First, try to get a list of available columns
            columns_query = f"""
                DESCRIBE SELECT * FROM read_parquet('azure://{CONTAINER_NAME}/{game_blob_name}')
            """
            columns_df = con.execute(columns_query).fetchdf()
            available_columns = columns_df['column_name'].tolist() if not columns_df.empty else []
            
            # Build a query based on available columns
            select_fields = []
            expected_columns = ['user_team', 'coach', 'away_team_name', 'event_date', 
                               'event_hour', 'event_minute', 'field_name', 'field_location', 
                               'field_type', 'field_temperature', 'game_status', 'my_team_ha']
            
            for col in expected_columns:
                if col in available_columns:
                    select_fields.append(col)
                else:
                    # Use a default value for missing columns
                    if col == 'user_team':
                        select_fields.append(f"'{team_id}' as user_team")
                    elif col in ['event_hour', 'event_minute']:
                        select_fields.append(f"0 as {col}")
                    else:
                        select_fields.append(f"'' as {col}")
            
            # Create and execute the query with only available columns
            game_query = f"""
                SELECT 
                    {", ".join(select_fields)}
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_blob_name}')
            """
            game_df = con.execute(game_query).fetchdf()
            
            # Handle the case where data might be empty
            if game_df.empty:
                return {
                    "status": "error",
                    "message": f"Game {game_id} exists but contains no data for team {team_id}",
                    "detail": "The game file exists but has no data. You may need to recreate this game."
                }
                
            game_data = game_df.to_dict(orient='records')[0]
            
            # Add game_id to the data
            game_data['game_id'] = game_id  # Add game_id from the request
            
        except Exception as e:
            logger.error(f"Error reading game data: {str(e)}")
            return {
                "status": "error",
                "message": f"Error reading game {game_id} for team {team_id}",
                "detail": f"There was an error reading the game data: {str(e)}"
            }
    except Exception as e:
        logger.error(f"Error getting game {game_id} for team {team_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"General error with game {game_id} for team {team_id}",
            "detail": f"There was a general error: {str(e)}"
        }
    
    return {
        "status": "success",
        "team_id": team_id,
        "game_id": game_id,  # Include game_id in the response
        "game_data": game_data
    }

@router.post("/{team_id}")
async def create_game(team_id: str, game_data: GameData):
    """
    Create a new game for a team with the next available ID.
    """
    return await create_or_update_game_internal(team_id, game_data)

@router.put("/{team_id}/{game_id}")
async def update_game(team_id: str, game_id: str, game_data: GameData):
    """
    Update an existing game for a team.
    """
    return await create_or_update_game_internal(team_id, game_data, game_id)

@router.delete("/{team_id}/{game_id}")
async def delete_game(team_id: str, game_id: str):
    """
    Delete a specific game for a team
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Define the blob name for the game
        game_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        game_blob_client = container_client.get_blob_client(game_blob_name)
        
        # Check if the game exists
        try:
            game_blob_client.get_blob_properties()
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found for team {team_id}: {str(e)}"
            )
        
        # Delete the game
        game_blob_client.delete_blob()
        
        # Also check for and delete associated roster files
        deleted_files = [game_blob_name]
        
        # Try to delete my roster file
        try:
            my_roster_blob_name = f"games/team_{team_id}/m_roster_{game_id}.parquet"
            my_roster_blob_client = container_client.get_blob_client(my_roster_blob_name)
            my_roster_blob_client.delete_blob()
            deleted_files.append(my_roster_blob_name)
        except Exception as e:
            # It's okay if the roster file doesn't exist
            logger.info(f"My roster file not found for game {game_id}, team {team_id}: {str(e)}")
        
        # Try to delete opponent roster file
        try:
            opponent_roster_blob_name = f"games/team_{team_id}/o_roster_{game_id}.parquet"
            opponent_roster_blob_client = container_client.get_blob_client(opponent_roster_blob_name)
            opponent_roster_blob_client.delete_blob()
            deleted_files.append(opponent_roster_blob_name)
        except Exception as e:
            # It's okay if the roster file doesn't exist
            logger.info(f"Opponent roster file not found for game {game_id}, team {team_id}: {str(e)}")
        
        return {
            "message": f"Game {game_id} deleted successfully for team {team_id}",
            "team_id": team_id,
            "game_id": game_id,
            "deleted_files": deleted_files
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting game {game_id} for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting game {game_id} for team {team_id}: {str(e)}"
        )

# Internal function to handle both creation and updates
async def create_or_update_game_internal(team_id: str, game_data: GameData, game_id: Optional[str] = None):
    """
    Internal function to handle both creating and updating games.
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # If game_id is not provided, find the next available game number
        if game_id is None:
            # List existing games to find the next game number
            games_prefix = f"games/team_{team_id}/game_"
            blobs = container_client.list_blobs(name_starts_with=games_prefix)
            
            game_numbers = []
            for blob in blobs:
                if blob.name.endswith('.parquet'):
                    try:
                        game_filename = blob.name.split('/')[-1]
                        game_num = int(game_filename.split('_')[1].split('.')[0])
                        game_numbers.append(game_num)
                    except:
                        continue
            
            next_game_number = 1
            if game_numbers:
                next_game_number = max(game_numbers) + 1
            
            game_id = str(next_game_number)
            operation = "created"
        else:
            # Check if the game exists
            game_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
            game_blob_client = container_client.get_blob_client(game_blob_name)
            
            try:
                # Check if the blob exists
                game_blob_client.get_blob_properties()
                operation = "updated"
            except Exception as e:
                # If the blob doesn't exist, we're creating a new game with a specified ID
                operation = "created"
        # Convert to DataFrame
        game_dict = game_data.model_dump()
        # Add game_id to the data
        game_dict['game_id'] = game_id
        df = pd.DataFrame([game_dict])
        print("------(*)(*)(*)(*)game info (*)(*)(*)(*) ------------")
        print(f"batter_df sample: {df.iloc[0].to_dict() if not df.empty else 'No data'}")
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the game data
        print(f"Uploading game data for team {team_id}, game {game_id}")
        game_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        game_blob_client = container_client.get_blob_client(game_blob_name)
        game_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Game {operation} successfully",
            "team_id": team_id,
            "game_id": game_id,
            "game_data": game_dict
        }
    except Exception as e:
        logger.error(f"Error creating/updating game for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating/updating game for team {team_id}: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/my_team_ha")
async def get_inning_scorebook(team_id: int, game_id: int):
    """
    Get my_team_ha for a specific game
    """
    try:
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        try:
            query = f"""
                SELECT                     
                    LOWER(my_team_ha) AS my_team_ha
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}')
            """
            my_team_ha_df = con.execute(query).fetchdf()
            if my_team_ha_df.empty:
                my_team_ha = "home"
                return my_team_ha
            else:
                print("no game data found")
                #I need to consistently format the string to 'home' or 'away' based upon the stored value.  
                if my_team_ha_df.iloc[0]['my_team_ha'] == "home":
                    print("returned home")
                    return 'home'
                else:
                    print("returned away")
                    return 'away'
        except Exception as e:
            logger.error(f"failure reading my_team_ha: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting my_team_ha for team {team_id}, game {game_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting my_team_ha for team {team_id}, game {game_id}: {str(e)}"
        )
