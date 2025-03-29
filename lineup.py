from utils import *  # Import all common utilities including the new function
import duckdb
from pydantic import field_validator, BaseModel, Field
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional, List, Union
import pandas as pd

router = APIRouter()

class LineupPlayer(BaseModel):
    jersey_number: str
    name: str
    position: str
    order_number: int
    inning_number: int
class Lineup(BaseModel):
    players: List[LineupPlayer]

class LineupItem(BaseModel):
    team_id: int
    game_id: int
    home_or_away: str
    inning_number: int
    order_number: int 
    jersey_number: str
    player_name: str




class LineupBatch(BaseModel):
    lineup: List[LineupItem]

@router.get("/{team_id}/{game_id}/{team_choice}/maximum-inning")
async def get_max_inning_number_lineup(team_id: str, game_id: str, team_choice: str):
    """
    Get the maximum inning number available for lineup data for a specific team, game, and team choice
    """
    try:
        # Get blob service client to list available innings
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        prefix = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_"
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        
        if not blobs:
            return {
                "max_inning": 0,
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "innings_available": False,
                "message": "No innings found for this team, game, and team choice"
            }
        
        # Extract inning numbers from blob names
        inning_numbers = []
        for blob in blobs:
            blob_name = blob.name
            inning_str = blob_name.replace(prefix, "").replace(".parquet", "")
            try:
                inning = int(inning_str)
                inning_numbers.append(inning)
            except ValueError:
                continue
        
        if not inning_numbers:
            return {
                "max_inning": 0,
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "innings_available": False,
                "message": "No valid inning numbers found in blob names"
            }
        
        max_inning = max(inning_numbers)
        
        return {
            "max_inning": max_inning,
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "innings_available": True,
            "available_innings": sorted(inning_numbers),
            "message": f"Maximum inning number is {max_inning}"
        }
    except Exception as e:
        logger.error(f"Error getting maximum inning number for lineup team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting maximum inning number for lineup: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/my")
async def get_my_lineup(team_id: str, game_id: str):
    """
    Get the lineup for the user's team for a specific game
    """
    try:
        # Get DuckDB connection
        con = get_duckdb_connection()
        # Get lineup data
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/m_lineup.parquet"
        try:
            lineup_query = f"""
                SELECT 
                    jersey_number,
                    name,
                    position,
                    order_number
                FROM read_parquet('azure://{CONTAINER_NAME}/{lineup_blob_name}')
                ORDER BY order_number
            """
            lineup_df = con.execute(lineup_query).fetchdf()
            
            # Handle the case where data might be empty
            if lineup_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Lineup not found for team {team_id}, game {game_id}"
                )
                
            lineup_data = lineup_df.to_dict(orient='records')
            
            return {
                "team_id": team_id,
                "game_id": game_id,
                "lineup": lineup_data
            }
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Lineup not found for team {team_id}, game {game_id}: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lineup for team {team_id}, game {game_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting lineup for team {team_id}, game {game_id}: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/opponent")
async def get_opponent_lineup(team_id: str, game_id: str):
    """
    Get the lineup for the opponent team for a specific game
    """
    try:
        # Get DuckDB connection
        con = get_duckdb_connection()
        
        # Get lineup data
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/o_lineup.parquet"
        try:
            lineup_query = f"""
                SELECT 
                    jersey_number,
                    name,
                    position,
                    order_number
                FROM read_parquet('azure://{CONTAINER_NAME}/{lineup_blob_name}')
                ORDER BY order_number
            """
            lineup_df = con.execute(lineup_query).fetchdf()
            
            # Handle the case where data might be empty
            if lineup_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Opponent lineup not found for team {team_id}, game {game_id}"
                )
                
            lineup_data = lineup_df.to_dict(orient='records')
            
            return {
                "team_id": team_id,
                "game_id": game_id,
                "lineup": lineup_data
            }
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Opponent lineup not found for team {team_id}, game {game_id}: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting opponent lineup for team {team_id}, game {game_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting opponent lineup for team {team_id}, game {game_id}: {str(e)}"
        )

@router.put("/{team_id}/{game_id}/my")
async def update_my_lineup(team_id: str, game_id: str, lineup: Lineup):
    """
    Update the lineup for the user's team for a specific game
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Create directory structure if it doesn't exist
        directory_path = f"games/team_{team_id}/game_{game_id}"
        
        # Convert to DataFrame
        lineup_dict = [player.model_dump() for player in lineup.players]
        df = pd.DataFrame(lineup_dict)
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the lineup data
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/m_lineup.parquet"
        lineup_blob_client = container_client.get_blob_client(lineup_blob_name)
        lineup_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": "Lineup updated successfully",
            "team_id": team_id,
            "game_id": game_id,
            "lineup": lineup_dict
        }
    except Exception as e:
        logger.error(f"Error updating lineup for team {team_id}, game {game_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating lineup for team {team_id}, game {game_id}: {str(e)}"
        )

@router.put("/{team_id}/{game_id}/opponent")
async def update_opponent_lineup(team_id: str, game_id: str, lineup: Lineup):
    """
    Update the lineup for the opponent team for a specific game
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Convert to DataFrame
        lineup_dict = [player.model_dump() for player in lineup.players]
        df = pd.DataFrame(lineup_dict)
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the lineup data
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/o_lineup.parquet"
        lineup_blob_client = container_client.get_blob_client(lineup_blob_name)
        lineup_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": "Opponent lineup updated successfully",
            "team_id": team_id,
            "game_id": game_id,
            "lineup": lineup_dict
        }
    except Exception as e:
        logger.error(f"Error updating opponent lineup for team {team_id}, game {game_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating opponent lineup for team {team_id}, game {game_id}: {str(e)}"
        )

@router.post("/{team_id}/{game_id}/{team_choice}/{inning_number}")
async def update_lineup_for_inning(team_id: str, game_id: str, team_choice: str, inning_number: int, lineup_data: Union[LineupBatch, List[LineupItem]]):
    """
    Update the lineup for a specific inning, using a URL that includes the inning number
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Handle both formats: array of LineupItems or LineupBatch object
        if isinstance(lineup_data, LineupBatch):
            # The standard format with lineup key
            items = lineup_data.lineup
        else:
            # Direct array format
            items = lineup_data
            
        # Validation
        if not items:
            raise HTTPException(
                status_code=400,
                detail="Lineup data is empty"
            )
            
        # Convert to list of dictionaries
        lineup_dict = [item.model_dump() for item in items]
        df = pd.DataFrame(lineup_dict)
        
        # Drop any duplicates, take the most recent entry
        df = df.drop_duplicates(subset=['order_number'], keep='last')
        
        # Force the inning_number to match the URL parameter
        df['inning_number'] = inning_number
        
        # Check if we have the minimum required fields
        required_fields = ['team_id', 'game_id', 'home_or_away', 'order_number', 'jersey_number', 'player_name']
        for field in required_fields:
            if field not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        # Validate team_id and game_id match URL parameters
        for item in items:
            if str(item.team_id) != team_id or str(item.game_id) != game_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"team_id and game_id in lineup data must match URL parameters"
                )
                
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the lineup data using the new path structure
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{inning_number}.parquet"
        lineup_blob_client = container_client.get_blob_client(lineup_blob_name)
        lineup_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Lineup updated successfully for {team_choice} team, inning {inning_number}",
            "team_id": int(team_id),
            "game_id": int(game_id),
            "inning_number": inning_number,
            "team_choice": team_choice
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating lineup for team {team_id}, game {game_id}, team_choice {team_choice}, inning {inning_number}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating lineup: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/{team_choice}")
async def get_lineup_data(team_id: str, game_id: str, team_choice: str, inning_number: Optional[int] = None):
    """
    Retrieve lineup data saved in parquet files for a specific team, game, and team choice
    """
    try:
        # Get DuckDB connection
        con = get_duckdb_connection()
        
        # If inning number is specified, get data for that inning only
        if inning_number is not None:
            lineup_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{inning_number}.parquet"
            try:
                # Read the specific parquet file
                query = f"""
                    SELECT 
                        CAST(team_id AS INTEGER) AS team_id,
                        CAST(game_id AS INTEGER) AS game_id,
                        CAST(inning_number AS INTEGER) AS inning_number,
                        CAST(order_number AS INTEGER) AS order_number,
                        jersey_number,
                        player_name,
                        CASE WHEN LOWER(home_or_away) = 'home' THEN 'home' ELSE 'away' END AS home_or_away
                    FROM read_parquet('azure://{CONTAINER_NAME}/{lineup_blob_name}')
                    ORDER BY order_number
                """
                lineup_df = con.execute(query).fetchdf()
                print("------(*)*)*)*) lineup retrieval ------------")
                print(f"batter_df sample: {lineup_df.iloc[0].to_dict() if not lineup_df.empty else 'No data'}")
                if lineup_df.empty:
                    lineup_available = 'no'
                    return {
                        "header": {
                            "team_id": int(team_id),
                            "game_id": int(game_id),
                            "lineup_available": lineup_available,
                            "home_or_away": "home" if team_choice == "home" else "away"
                        },
                    }
                
                # Format to desired structure
                inning_data = lineup_df[['order_number', 'jersey_number', 'player_name']].to_dict(orient='records')
                
                return {
                    "header": {
                        "team_id": int(team_id),
                        "game_id": int(game_id),
                        "lineup_available": lineup_available,
                        "home_or_away": "home" if team_choice == "home" else "away"
                    },
                    "innings_data": {
                        str(inning_number): inning_data
                    }
                }
            except Exception as e:
                logger.error(f"Error getting lineup for inning {inning_number}: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Lineup not found for inning {inning_number}: {str(e)}"
                )
        
        # If inning number is not specified, use Azure Blob storage to list available innings
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        prefix = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_"
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        
        if not blobs:
            raise HTTPException(
                status_code=404,
                detail=f"No lineup data found for team {team_id}, game {game_id}, team choice {team_choice}"
            )
        
        # Extract inning numbers from blob names
        innings = []
        inning_to_blob = {}
        
        for blob in blobs:
            blob_name = blob.name
            inning_str = blob_name.replace(prefix, "").replace(".parquet", "")
            
            try:
                inning = int(inning_str)
                innings.append(inning)
                inning_to_blob[inning] = blob_name
            except ValueError:
                # Skip if inning number is not a valid integer
                continue
        
        # Sort innings
        sorted_innings = sorted(innings)
        
        if not sorted_innings:
            return {
                "header": {
                    "team_id": int(team_id),
                    "game_id": int(game_id),
                    "home_or_away": "home" if team_choice == "home" else "away"
                },
                "innings_data": {}
            }
        
        # Build innings_data dictionary by reading each inning's data
        innings_data = {}
        
        for inning in sorted_innings:
            blob_name = inning_to_blob[inning]
            try:
                query = f"""
                    SELECT 
                        CAST(order_number AS INTEGER) AS order_number,
                        jersey_number,
                        player_name
                    FROM read_parquet('azure://{CONTAINER_NAME}/{blob_name}')
                    ORDER BY order_number
                """
                inning_df = con.execute(query).fetchdf()
                
                if not inning_df.empty:
                    # Add data for this inning to the innings_data dictionary
                    innings_data[str(inning)] = inning_df.to_dict(orient='records')
                    innings_data_available = 'yes'
                if inning_df.empty:
                    innings_data_available = 'no'
            except Exception as e:
                logger.error(f"Error reading inning {inning} data: {str(e)}")
                # Continue with next inning if one fails
                continue
            
        return {
            "header": {
                "team_id": int(team_id),
                "game_id": int(game_id),
                "home_or_away": "home" if team_choice == "home" else "away"
            },
            "innings_data": innings_data,
            "innings_data_available": innings_data_available
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving lineup data for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving lineup data: {str(e)}"
        )

@router.delete("/{team_id}/{game_id}/{team_choice}/{inning_number}")
async def delete_lineup_data(team_id: str, game_id: str, team_choice: str, inning_number: int):
    """
    Delete lineup data for a specific team, game, team choice, and inning
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Construct the blob name
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{inning_number}.parquet"
        blob_client = container_client.get_blob_client(lineup_blob_name)
        
        # Check if the blob exists
        exists = False
        try:
            blob_properties = blob_client.get_blob_properties()
            exists = True
        except Exception:
            exists = False
            
        if not exists:
            return {
                "message": f"No lineup data found for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}",
                "deleted": False,
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "inning_number": inning_number
            }
        
        # Delete the blob
        blob_client.delete_blob()
        
        return {
            "message": f"Lineup data deleted successfully for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}",
            "deleted": True,
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "inning_number": inning_number
        }
    except Exception as e:
        logger.error(f"Error deleting lineup data for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting lineup data: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/{team_choice}/{inning_number}/new_inning")
async def get_new_inning_lineup(team_id: str, game_id: str, team_choice: str, inning_number: int):
    """
    Retrieve lineup data for creating a new inning based on the previous inning.
    Returns the specified inning's lineup with only the basic fields needed for a new inning.
    """
    try:
        # Get DuckDB connection
        con = get_duckdb_connection()
        
        # Construct the blob name for the requested inning
        lineup_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{inning_number}.parquet"
        
        try:
            # Read the specific parquet file
            query = f"""
                SELECT 
                    CAST(team_id AS INTEGER) AS team_id,
                    CAST(game_id AS INTEGER) AS game_id,
                    CAST(order_number AS INTEGER) AS order_number,
                    jersey_number,
                    player_name
                FROM read_parquet('azure://{CONTAINER_NAME}/{lineup_blob_name}')
                ORDER BY order_number
            """
            lineup_df = con.execute(query).fetchdf()
            
            if lineup_df.empty:
                # If requested inning is not found, try to find the most recent inning
                blob_service_client = get_blob_service_client()
                container_client = blob_service_client.get_container_client(CONTAINER_NAME)
                
                prefix = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_"
                blobs = list(container_client.list_blobs(name_starts_with=prefix))
                
                if not blobs:
                    return {
                        "message": f"No lineup data found for team {team_id}, game {game_id}, team choice {team_choice}",
                        "team_id": int(team_id),
                        "game_id": int(game_id),
                        "team_choice": team_choice,
                        "source_inning": None,
                        "lineup": []
                    }
                
                # Find the most recent inning
                innings = []
                for blob in blobs:
                    blob_name = blob.name
                    inning_str = blob_name.replace(prefix, "").replace(".parquet", "")
                    try:
                        innings.append(int(inning_str))
                    except ValueError:
                        continue
                
                if not innings:
                    return {
                        "message": f"No valid inning data found for team {team_id}, game {game_id}, team choice {team_choice}",
                        "team_id": int(team_id),
                        "game_id": int(game_id),
                        "team_choice": team_choice,
                        "source_inning": None,
                        "lineup": []
                    }
                
                # Get the most recent inning
                most_recent_inning = max(innings)
                source_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{most_recent_inning}.parquet"
                
                # Read the most recent inning data
                query = f"""
                    SELECT 
                        CAST(team_id AS INTEGER) AS team_id,
                        CAST(game_id AS INTEGER) AS game_id,
                        CAST(order_number AS INTEGER) AS order_number,
                        jersey_number,
                        player_name
                    FROM read_parquet('azure://{CONTAINER_NAME}/{source_blob_name}')
                    WHERE inning_number = (SELECT MAX(inning_number) as inning_number 
                    FROM read_parquet('azure://{CONTAINER_NAME}/{source_blob_name}'))
                    ORDER BY order_number
                """
                lineup_df = con.execute(query).fetchdf()
                
                if lineup_df.empty:
                    return {
                        "message": f"No lineup data found in the most recent inning ({most_recent_inning}) for team {team_id}, game {game_id}, team choice {team_choice}",
                        "team_id": int(team_id),
                        "game_id": int(game_id),
                        "team_choice": team_choice,
                        "source_inning": most_recent_inning,
                        "lineup": []
                    }
                
                source_inning = most_recent_inning
            else:
                source_inning = inning_number
            
            # Format the data for the new inning
            lineup_data = lineup_df.to_dict(orient='records')
            
            return {
                "message": f"Lineup template for new inning based on inning {source_inning}",
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "source_inning": source_inning,
                "lineup": lineup_data
            }
        except Exception as e:
            logger.error(f"Error getting lineup template for new inning: {str(e)}")
            return {
                "message": f"Error retrieving lineup data: {str(e)}",
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "source_inning": None,
                "lineup": []
            }
            
    except Exception as e:
        logger.error(f"Error preparing new inning lineup for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        return {
            "message": f"Error preparing new inning lineup: {str(e)}",
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "source_inning": None,
            "lineup": []
        }

@router.post("/{team_id}/{game_id}/{team_choice}/copy/{from_inning}/{to_inning}")
async def copy_lineup_data(team_id: str, game_id: str, team_choice: str, from_inning: int, to_inning: int):
    """
    Copy lineup data from one inning to another for a specific team and game
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Construct the source and destination blob names
        source_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{from_inning}.parquet"
        destination_blob_name = f"games/team_{team_id}/game_{game_id}/lineup/{team_choice}_offense_{to_inning}.parquet"
        
        # Check if the source blob exists
        source_blob_client = container_client.get_blob_client(source_blob_name)
        source_exists = False
        try:
            source_blob_properties = source_blob_client.get_blob_properties()
            source_exists = True
        except Exception:
            source_exists = False
            
        if not source_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Source inning lineup data not found for team {team_id}, game {game_id}, team choice {team_choice}, inning {from_inning}"
            )
        
        # Check if the destination already exists
        destination_blob_client = container_client.get_blob_client(destination_blob_name)
        destination_exists = False
        try:
            destination_blob_properties = destination_blob_client.get_blob_properties()
            destination_exists = True
        except Exception:
            destination_exists = False
        
        # Get the data from the source blob
        con = get_duckdb_connection()
        query = f"""
            SELECT 
                CAST(team_id AS INTEGER) AS team_id,
                CAST(game_id AS INTEGER) AS game_id,
                home_or_away,
                CAST(order_number AS INTEGER) AS order_number,
                jersey_number,
                player_name
            FROM read_parquet('azure://{CONTAINER_NAME}/{source_blob_name}')
        """
        source_df = con.execute(query).fetchdf()
        
        if source_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Source inning lineup data is empty for team {team_id}, game {game_id}, team choice {team_choice}, inning {from_inning}"
            )
        
        # Update the inning number to the destination inning
        source_df['inning_number'] = to_inning
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        source_df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload to the destination blob
        destination_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Lineup data copied successfully from inning {from_inning} to inning {to_inning}",
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "from_inning": from_inning,
            "to_inning": to_inning,
            "lineup_count": len(source_df),
            "overwritten": destination_exists
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error copying lineup data from inning {from_inning} to {to_inning} for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error copying lineup data: {str(e)}"
        ) 