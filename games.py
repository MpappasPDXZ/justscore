from utils import *  # Import all common utilities including the new function
import duckdb
from pydantic import field_validator
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional
import pandas as pd
import json

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
    field_temperature: int
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
# Utility function to safely convert parameters to integers when possible
def safe_int_conversion(value):
    """
    Safely convert a value to integer if it's a digit string.
    Otherwise, return the original value.
    """
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value

@router.get("/{team_id}/gamelist") 
async def list_games(team_id: str):
    """
    List all games for a specific team by finding game_[number].parquet files
    """
    try:
        con = get_duckdb_connection()
        # Convert team_id, game_id, inning_number, and batter_seq_id to integers if possible
        team_id_val = safe_int_conversion(team_id)
        try:
            query = f"""
                WITH game_data AS (
                    SELECT 
                        *
                    FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_*.parquet', union_by_name=true)
                ),
                sorted_game_data AS (
                    SELECT *
                    FROM game_data
                    ORDER BY 
                        -- First try to sort by year, which is at positions 7-10 in MM-DD-YYYY format
                        CASE WHEN LENGTH(event_date) >= 10 THEN SUBSTRING(event_date, 7, 4) ELSE '0000' END DESC,
                        -- Then by month (positions 1-2)
                        CASE WHEN LENGTH(event_date) >= 5 THEN SUBSTRING(event_date, 1, 2) ELSE '00' END DESC,
                        -- Finally by day (positions 4-5)
                        CASE WHEN LENGTH(event_date) >= 5 THEN SUBSTRING(event_date, 4, 2) ELSE '00' END DESC,
                        -- Also include hour and minute for events on the same day
                        event_hour DESC,
                        event_minute DESC
                ),
                game_count AS (
                    SELECT COUNT(*) as count FROM game_data
                )
                SELECT 
                    {team_id_val} as team_id,
                    (SELECT count FROM game_count) as games_count,
                    COALESCE(
                        json_group_array(
                            json_object(
                                'game_id', game_id,
                                'user_team', user_team,
                                'coach', coach,
                                'away_team_name', away_team_name,
                                'event_date', event_date,
                                'event_hour', event_hour,
                                'event_minute', event_minute,
                                'field_name', field_name,
                                'field_location', field_location,
                                'field_type', field_type,
                                'field_temperature', field_temperature,
                                'game_status', game_status,
                                'my_team_ha', my_team_ha
                            )
                        ),
                        '[]'
                    ) as games
                FROM sorted_game_data
            """
            result = con.execute(query).fetchone()

            if result is None:
                logger.warning(f"No games found for team {team_id}")
                return {
                    "team_id": team_id,
                    "games_count": 0,
                    "games": []
                }
                
            return {
                "team_id": result[0],
                "games_count": result[1],
                "games": json.loads(result[2])
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
    except Exception as e:
        logger.error(f"Error listing games for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing games for team {team_id}: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/get_one_game")
async def get_game(team_id: str, game_id: str):
    """
    Get details for a specific game
    """
    try:
        # Step 1: Create DuckDB connection
        con = get_duckdb_connection()
        
        # Step 2: Convert strings to integers
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        
        # Step 3: Execute DuckDB query with JSON object creation
        query = f"""
            WITH game_data AS (
                SELECT 
                    *
                FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_{game_id_val}.parquet')
            )
            SELECT 
                json_object(
                    'user_team', user_team,
                    'coach', coach,
                    'away_team_name', away_team_name,
                    'event_date', event_date,
                    'event_hour', event_hour,
                    'event_minute', event_minute,
                    'field_name', field_name,
                    'field_location', field_location,
                    'field_type', field_type,
                    'field_temperature', field_temperature,
                    'game_status', game_status,
                    'my_team_ha', my_team_ha
                ) as game_data
            FROM game_data
        """
        
        # Step 4: Get the result
        result = con.execute(query).fetchone()
        
        if result is None:
            logger.warning(f"No game found for team {team_id_val}, game {game_id_val}")
            return {
                "team_id": team_id_val,
                "game_id": game_id_val,
                "game_data": None
            }
        
        # Step 5: Pydantic validation
        try:
            game_data = GameData(**json.loads(result[0]))
        except Exception as e:
            logger.error(f"Error validating game data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error validating game data: {str(e)}"
            )
        
        # Step 6: Return the response with static values
        return {
            "team_id": team_id_val,
            "game_id": game_id_val,
            "game_data": game_data.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error getting game {game_id_val} for team {team_id_val}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting game {game_id_val} for team {team_id_val}: {str(e)}"
        )

@router.delete("/{team_id}/{game_id}/delete_game")
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

    except Exception as e:
        logger.error(f"Error deleting game {game_id} for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting game {game_id} for team {team_id}: {str(e)}"
        )
@router.post("/{team_id}/create_game")
async def create_game(team_id: str, game_data: GameData):
    """
    Create a new game for a team
    """
    try:
        # Convert team_id to integer
        team_id_val = safe_int_conversion(team_id)
        
        # Try to find the maximum game_id
        con = get_duckdb_connection()
        try:
            query = f"""
                SELECT MAX(game_id) as max_game_id
                FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_*.parquet')
            """
            result = con.execute(query).fetchone()
            max_game_id = result[0] if result and result[0] is not None else 0
            new_game_id = max_game_id + 1
        except Exception as e:
            # If there are no games for this team yet, start with game_id 1
            new_game_id = 1
            logger.info(f"No existing games found for team {team_id_val}, starting with game_id 1")
        finally:
            # Close the DuckDB connection
            con.close()
        
        # Validate the game data
        try:
            game_data = GameData(**game_data.model_dump())
        except Exception as e:
            logger.error(f"Error validating game data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error validating game data: {str(e)}"
            )
        
        # Create the game file
        game_blob_name = f"games/team_{team_id_val}/game_{new_game_id}.parquet"
        container_client = get_blob_service_client().get_container_client(CONTAINER_NAME)
        game_blob_client = container_client.get_blob_client(game_blob_name)
        
        # Convert the game data to a DataFrame and then to a parquet file
        game_dict = game_data.model_dump()
        game_dict['game_id'] = new_game_id  # Add game_id to the data
        df = pd.DataFrame([game_dict])
        
        # Convert DataFrame to parquet bytes
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload the parquet file to Azure Blob Storage
        game_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "team_id": team_id_val,
            "game_id": new_game_id,
            "message": f"Game created successfully for team {team_id_val} and game {new_game_id}"
        }
    except Exception as e:
        logger.error(f"Error creating game for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating game for team {team_id}: {str(e)}"
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

@router.put("/{team_id}/{game_id}/edit_game")
async def edit_game(team_id: str, game_id: str, game_data: GameData):
    """
    Edit an existing game for a team
    """
    try:
        # Convert team_id and game_id to integers
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        
        # Check if the game exists
        con = get_duckdb_connection()
        try:
            query = f"""
                SELECT COUNT(*) as count
                FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_{game_id_val}.parquet')
            """
            result = con.execute(query).fetchone()
            if result is None or result[0] == 0:
                logger.warning(f"Game {game_id_val} not found for team {team_id_val}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Game {game_id_val} not found for team {team_id_val}"
                )
        except Exception as e:
            error_msg = str(e)
            if "Failed to open" in error_msg or "No files found" in error_msg:
                logger.warning(f"Game {game_id_val} not found for team {team_id_val}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Game {game_id_val} not found for team {team_id_val}"
                )
            else:
                logger.error(f"Error checking if game exists: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error checking if game exists: {str(e)}"
                )
        finally:
            con.close()
        
        # Validate the game data
        try:
            game_data = GameData(**game_data.model_dump())
        except Exception as e:
            logger.error(f"Error validating game data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error validating game data: {str(e)}"
            )
        
        # Update the game file
        game_blob_name = f"games/team_{team_id_val}/game_{game_id_val}.parquet"
        container_client = get_blob_service_client().get_container_client(CONTAINER_NAME)
        game_blob_client = container_client.get_blob_client(game_blob_name)
        
        # Convert the game data to a DataFrame and then to a parquet file
        game_dict = game_data.model_dump()
        game_dict['game_id'] = game_id_val  # Add game_id to the data
        df = pd.DataFrame([game_dict])
        
        # Convert DataFrame to parquet bytes
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload the parquet file to Azure Blob Storage
        game_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "team_id": team_id_val,
            "game_id": game_id_val,
            "message": f"Game updated successfully for team {team_id_val} and game {game_id_val}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating game for team {team_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating game for team {team_id}: {str(e)}"
        )
