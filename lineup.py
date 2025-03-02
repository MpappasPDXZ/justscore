from utils import *  # Import all common utilities including the new function
import duckdb
from pydantic import field_validator
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional
import pandas as pd

router = APIRouter()

class LineupPlayer(BaseModel):
    jersey_number: str
    name: str
    position: str
    order_number: int

class Lineup(BaseModel):
    players: List[LineupPlayer]

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