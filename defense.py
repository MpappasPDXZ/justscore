from utils import *  # Import all common utilities
import duckdb
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional, List, Union

router = APIRouter()

class DefensePosition(BaseModel):
    team_id: int
    game_id: int
    home_or_away: str
    inning_number: int
    position_number: int
    jersey_number: str
    player_name: str
    batter_seq_id: int
    batter_seq_id_to: int

class DefenseBatch(BaseModel):
    positions: List[DefensePosition]

@router.post("/{team_id}/{game_id}/{team_choice}/{inning_number}")
async def update_defense_positions(team_id: str, game_id: str, team_choice: str, inning_number: int, 
                                   defense_data: Union[DefenseBatch, List[DefensePosition]]):
    """
    Update defensive positions for a specific team, game, inning
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Handle both formats: array of DefensePosition or DefenseBatch object
        if isinstance(defense_data, DefenseBatch):
            # The standard format with positions key
            items = defense_data.positions
        else:
            # Direct array format
            items = defense_data
            
        # Validation
        if not items:
            raise HTTPException(
                status_code=400,
                detail="Defense positions data is empty"
            )
            
        # Convert to list of dictionaries
        positions_dict = [item.model_dump() for item in items]
        df = pd.DataFrame(positions_dict)
        
        # Force the inning_number, team_id, game_id to match URL parameters
        df['inning_number'] = inning_number
        df['team_id'] = int(team_id)
        df['game_id'] = int(game_id)
        
        # Check if we have the minimum required fields
        required_fields = ['team_id', 'game_id', 'home_or_away', 'inning_number', 
                          'position_number', 'jersey_number', 'player_name', 
                          'batter_seq_id', 'batter_seq_id_to']
        for field in required_fields:
            if field not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Validate home_or_away matches team_choice
        expected_home_or_away = "home" if team_choice == "home" else "away"
        if not all(df['home_or_away'] == expected_home_or_away):
            # Force it to match
            df['home_or_away'] = expected_home_or_away
                
        # Validate that the batter sequence ranges don't overlap for the same position
        # (except for position 0 which is bench and can have multiple players)
        validation_issues = []
        
        # Group by position_number (for non-bench positions)
        for position, group in df[df['position_number'] != 0].groupby('position_number'):
            if len(group) <= 1:
                continue  # No need to check for overlaps with just one player
                
            # Sort by batter_seq_id for easier comparison
            sorted_group = group.sort_values('batter_seq_id')
            
            # Convert to records for easier iteration
            records = sorted_group.to_dict('records')
            
            # Check for overlapping ranges
            for i in range(len(records) - 1):
                current = records[i]
                next_record = records[i + 1]
                
                # Check if current player's range overlaps with next player's range
                if current['batter_seq_id_to'] >= next_record['batter_seq_id']:
                    validation_issues.append(
                        f"Position {position}: Player {current['player_name']} (batters {current['batter_seq_id']}-{current['batter_seq_id_to']}) " +
                        f"overlaps with {next_record['player_name']} (batters {next_record['batter_seq_id']}-{next_record['batter_seq_id_to']})"
                    )
        
        if validation_issues:
            raise HTTPException(
                status_code=400,
                detail=f"Validation issues with defensive positions: {'; '.join(validation_issues)}"
            )
        
        # Make sure each field position (1-9) has coverage for all batters in inning
        active_positions = set(df[df['position_number'] != 0]['position_number'].unique())
        expected_positions = set(range(1, 10))  # Positions 1-9
        missing_positions = expected_positions - active_positions
        
        if missing_positions and len(df) > 0:
            logger.warning(f"Missing defensive positions: {missing_positions} for team {team_id}, game {game_id}, inning {inning_number}")
        
        # Convert DataFrame to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Upload the defense data
        defense_blob_name = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_{inning_number}.parquet"
        defense_blob_client = container_client.get_blob_client(defense_blob_name)
        defense_blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
        
        return {
            "message": f"Defense positions updated successfully for {team_choice} team, inning {inning_number}",
            "team_id": int(team_id),
            "game_id": int(game_id),
            "inning_number": inning_number,
            "team_choice": team_choice,
            "positions_count": len(df),
            "missing_positions": list(missing_positions) if missing_positions else []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating defense positions for team {team_id}, game {game_id}, team_choice {team_choice}, inning {inning_number}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating defense positions: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/{team_choice}/maximum-inning")
async def get_max_inning_number(team_id: str, game_id: str, team_choice: str):
    """
    Get the maximum inning number available for a specific team, game, and team choice
    """
    try:
        # Get blob service client to list available innings
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        prefix = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_"
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
        logger.error(f"Error getting maximum inning number for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting maximum inning number: {str(e)}"
        )

@router.get("/{team_id}/{game_id}/{team_choice}/{inning_number}")
async def get_defense_positions(team_id: str, game_id: str, team_choice: str, inning_number: int, batter_seq_id: Optional[int] = None):
    """
    Retrieve defensive positions for a specific team, game, inning, and optionally filtered by batter_seq_id
    """
    try:
        # Get DuckDB connection
        con = get_duckdb_connection()
        
        # Construct the blob name
        defense_blob_name = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_{inning_number}.parquet"
        
        try:
            # Build the query with optional batter_seq_id filter
            if batter_seq_id is not None:
                query = f"""
                    SELECT 
                        CAST(team_id AS INTEGER) AS team_id,
                        CAST(game_id AS INTEGER) AS game_id,
                        CAST(inning_number AS INTEGER) AS inning_number,
                        CAST(position_number AS INTEGER) AS position_number,
                        jersey_number,
                        player_name,
                        CAST(batter_seq_id AS INTEGER) AS batter_seq_id,
                        CAST(batter_seq_id_to AS INTEGER) AS batter_seq_id_to,
                        home_or_away
                    FROM read_parquet('azure://{CONTAINER_NAME}/{defense_blob_name}')
                    WHERE {batter_seq_id} >= batter_seq_id AND {batter_seq_id} <= batter_seq_id_to
                    ORDER BY position_number
                """
            else:
                query = f"""
                    SELECT 
                        CAST(team_id AS INTEGER) AS team_id,
                        CAST(game_id AS INTEGER) AS game_id,
                        CAST(inning_number AS INTEGER) AS inning_number,
                        CAST(position_number AS INTEGER) AS position_number,
                        jersey_number,
                        player_name,
                        CAST(batter_seq_id AS INTEGER) AS batter_seq_id,
                        CAST(batter_seq_id_to AS INTEGER) AS batter_seq_id_to,
                        home_or_away
                    FROM read_parquet('azure://{CONTAINER_NAME}/{defense_blob_name}')
                    ORDER BY position_number, batter_seq_id
                """
            
            defense_df = con.execute(query).fetchdf()
            
            if defense_df.empty:
                return {
                    "defense_available": "no",
                    "innings_data": {}
                }
            
            # Format the response
            positions_data = defense_df.to_dict(orient='records')
            
            # Structure the response with inning_number as key
            return {
                "defense_available": "yes",
                "innings_data": {
                    str(inning_number): positions_data
                }
            }
        except Exception as e:
            logger.error(f"Error getting defense positions: {str(e)}")
            return {
                "defense_available": "no",
                "innings_data": {}
            }
    except Exception as e:
        logger.error(f"Error retrieving defense positions for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}: {str(e)}")
        return {
            "defense_available": "no",
            "innings_data": {}
        }

@router.get("/{team_id}/{game_id}/{team_choice}")
async def get_all_defense_positions(team_id: str, game_id: str, team_choice: str):
    """
    Retrieve defensive positions for all innings for a specific team and game
    """
    try:
        # Get blob service client to list available innings
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        prefix = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_"
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        
        if not blobs:
            return {
                "defense_available": "no",
                "innings_data": {}
            }
        
        # Extract inning numbers from blob names
        inning_to_blob = {}
        for blob in blobs:
            blob_name = blob.name
            inning_str = blob_name.replace(prefix, "").replace(".parquet", "")
            try:
                inning = int(inning_str)
                inning_to_blob[inning] = blob_name
            except ValueError:
                continue
        
        if not inning_to_blob:
            return {
                "defense_available": "no",
                "innings_data": {}
            }
        
        # Get DuckDB connection
        con = get_duckdb_connection()
        
        # Build innings_data dictionary by reading each inning's data
        innings_data = {}
        defense_available = "no"
        
        for inning, blob_name in inning_to_blob.items():
            try:
                query = f"""
                    SELECT 
                        CAST(team_id AS INTEGER) AS team_id,
                        CAST(game_id AS INTEGER) AS game_id,
                        CAST(inning_number AS INTEGER) AS inning_number,
                        CAST(position_number AS INTEGER) AS position_number,
                        jersey_number,
                        player_name,
                        CAST(batter_seq_id AS INTEGER) AS batter_seq_id,
                        CAST(batter_seq_id_to AS INTEGER) AS batter_seq_id_to,
                        home_or_away
                    FROM read_parquet('azure://{CONTAINER_NAME}/{blob_name}')
                    ORDER BY position_number, batter_seq_id
                """
                inning_df = con.execute(query).fetchdf()
                
                if not inning_df.empty:
                    innings_data[str(inning)] = inning_df.to_dict(orient='records')
                    defense_available = "yes"
            except Exception as e:
                logger.error(f"Error reading inning {inning} defense data: {str(e)}")
                continue
        
        return {
            "defense_available": defense_available,
            "innings_data": innings_data
        }
    except Exception as e:
        logger.error(f"Error retrieving all defense positions for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        return {
            "defense_available": "no",
            "innings_data": {}
        }

@router.delete("/{team_id}/{game_id}/{team_choice}/{inning_number}")
async def delete_defense_positions(team_id: str, game_id: str, team_choice: str, inning_number: int):
    """
    Delete defensive positions for a specific team, game, and inning
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Construct the blob name
        defense_blob_name = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_{inning_number}.parquet"
        blob_client = container_client.get_blob_client(defense_blob_name)
        
        # Check if the blob exists
        exists = False
        try:
            blob_properties = blob_client.get_blob_properties()
            exists = True
        except Exception:
            exists = False
            
        if not exists:
            return {
                "message": f"No defense positions found for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}",
                "deleted": False,
                "team_id": int(team_id),
                "game_id": int(game_id),
                "team_choice": team_choice,
                "inning_number": inning_number
            }
        
        # Delete the blob
        blob_client.delete_blob()
        
        return {
            "message": f"Defense positions deleted successfully for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}",
            "deleted": True,
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "inning_number": inning_number
        }
    except Exception as e:
        logger.error(f"Error deleting defense positions for team {team_id}, game {game_id}, team choice {team_choice}, inning {inning_number}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting defense positions: {str(e)}"
        )

@router.post("/{team_id}/{game_id}/{team_choice}/copy/{from_inning}/{to_inning}")
async def copy_defense_positions(team_id: str, game_id: str, team_choice: str, from_inning: int, to_inning: int):
    """
    Copy defensive positions from one inning to another for a specific team and game
    """
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Construct the source and destination blob names
        source_blob_name = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_{from_inning}.parquet"
        destination_blob_name = f"games/team_{team_id}/game_{game_id}/defense/{team_choice}_defense_{to_inning}.parquet"
        
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
                detail=f"Source inning defense positions not found for team {team_id}, game {game_id}, team choice {team_choice}, inning {from_inning}"
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
                CAST(position_number AS INTEGER) AS position_number,
                jersey_number,
                player_name,
                CAST(batter_seq_id AS INTEGER) AS batter_seq_id,
                CAST(batter_seq_id_to AS INTEGER) AS batter_seq_id_to
            FROM read_parquet('azure://{CONTAINER_NAME}/{source_blob_name}')
        """
        source_df = con.execute(query).fetchdf()
        
        if source_df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Source inning defense positions are empty for team {team_id}, game {game_id}, team choice {team_choice}, inning {from_inning}"
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
            "message": f"Defense positions copied successfully from inning {from_inning} to inning {to_inning}",
            "team_id": int(team_id),
            "game_id": int(game_id),
            "team_choice": team_choice,
            "from_inning": from_inning,
            "to_inning": to_inning,
            "positions_count": len(source_df),
            "overwritten": destination_exists
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error copying defense positions from inning {from_inning} to {to_inning} for team {team_id}, game {game_id}, team choice {team_choice}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error copying defense positions: {str(e)}"
        ) 