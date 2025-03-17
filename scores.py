from utils import *  # Import all common utilities
import duckdb
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Body, Path, Query
from io import BytesIO
from typing import Optional, List
import pandas as pd
import requests
import numpy as np
import json

router = APIRouter()

# Add this list at the top of your file with other imports
ARRAY_FIELDS = ["hit_around_bases", "stolen_bases", "pa_error_on", "br_error_on"]

class ScoreData(BaseModel):
    gameId: int
    InningN: int
    h_a: str  # "home" or "away"
    my_team_ha: str  # "home" or "away"
    batter_seq_id: int
    batter_jersey_number: str
    pitcher_jersey_number: str
    balls_before_play: int
    strikes_before_play: int
    fouls_after_two_strikes: int
    pa_result: str  # BB,1B,2B,3B,HR,HB, K, KK, E, FO
    detailed_result: str
    hard_hit: int  # 1 for yes, 0 for no
    bunt_or_slap: int  # 1 for bunt, 2 for slap, 0 for no
    base_running: str
    base_running_stolen_base: int

class InningScores(BaseModel):
    team_id: str
    game_id: str
    inning_number: int
    scores: List[ScoreData]

class PlateAppearanceData(BaseModel):
    teamId: str
    gameId: str
    inning_number: int
    home_or_away: str
    my_team_ha: str
    order_number: int
    batter_jersey_number: str
    batter_name: str
    batter_seq_id: int
    out: int
    out_at: int
    balls_before_play: int
    pitch_count: int
    wild_pitches: Optional[int] = 0
    wild_pitch: Optional[int] = 0
    passed_ball: int
    strikes_before_play: int
    strikes_watching: int
    strikes_swinging: int
    strikes_unsure: int
    ball_swinging: int
    fouls: int
    pa_result: str
    detailed_result: Optional[str] = ""
    base_running: Optional[str] = ""
    hit_to: str
    pa_why: str
    pa_error_on: str
    br_result: int
    br_stolen_bases: int
    br_error_on: Optional[str] = "0"
    base_running_hit_around: int
    base_running_other: int
    bases_reached: Optional[str] = "0"
    hard_hit: Optional[int] = 0
    bunt_or_slap: Optional[int] = 0
    base_running_stolen_base: Optional[int] = 0
    stolen_bases: List[int] = []
    hit_around_bases: List[int] = []

@router.get("/{team_id}/{game_id}/{inning_number}/{team_choice}/{my_team_ha}")
async def get_inning_scorebook(team_id: str, game_id: str, inning_number: int, team_choice: str, my_team_ha: str):
    """
    Get detailed scorebook-style data for a specific inning, joining batting results with lineup information
    """
    try:
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        # Step 1: get the lineup info
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        # I need this to find the file for the lineup name
        if (my_team_ha == "home" and team_choice == "home") or (my_team_ha == "away" and team_choice == "away"):
            team_data_key = 'm'
        else:
            team_data_key = 'o'
        # Get lineup info with duckdb
        lineup_info_blob_name = f"games/team_{team_id}/game_{game_id}/{team_data_key}_lineup.parquet"
        try:
            query = f"""
                SELECT                     
                    jersey_number,
                    name,
                    position,
                    order_number 
                FROM read_parquet('azure://{CONTAINER_NAME}/{lineup_info_blob_name}')
                ORDER BY order_number
            """
            lineup_df = con.execute(query).fetchdf()
            if lineup_df.empty:
                print("no lineup data found")
                lineup_available = False
                lineup_entries = []
            else:
                lineup_available = True
                lineup_entries = lineup_df.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Error reading lineup info: {str(e)}")
            lineup_available = False
            lineup_entries = []
            
        # Step 3: Get at bat data from the new file structure
        pa_info_blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_*.parquet"
        batter_df = pd.DataFrame()  # Initialize empty DataFrame

        try:
            query = f"""
                SELECT                        
                    order_number, -- lineup order int
                    batter_seq_id, -- int that resets at the top of each inning
                    inning_number, -- 1 through 9
                    home_or_away, -- "home" or "away"
                    batting_order_position, -- int 1 through 20
                    team_id, -- int that is the primary key to my team
                    teamId, -- int that is used for state management
                    game_id,  -- int that is used for fast game parsing within a team            
                    gameId, -- int that is used for state management
                    out, -- int 0 or 1
                    my_team_ha, -- "home" or "away"
                    --one off tracking
                    rbi, -- int or null
                    passed_ball, -- int or null
                    wild_pitch, -- int or null
                    qab, -- 'QAB' or null
                    hard_hit, -- 'HH' or null
                    late_swings, -- int or null
                    --one off tracking
                    --at the plate
                    out_at, -- int (0-4)
                    pa_why, --str
                    pa_result, --int (0-4)
                    hit_to, --str
                    pa_error_on, --list of ints
                    --at the plate
                    --base running
                    br_result, -- int or null
                    br_stolen_bases, --list of ints
                    base_running_hit_around, --list of ints
                    br_error_on, --list of ints
                    --balls and strikes
                    pitch_count, -- need to add one to this and include fouls properly
                    balls_before_play, --int 0,1,2,3
                    strikes_before_play, --int 0,1,2
                    strikes_unsure, -- int 0, 1, 2
                    strikes_watching, -- int 0, 1, 2
                    strikes_swinging, -- int 0, 1, 2
                    ball_swinging, -- int 0, 1, 2
                    fouls, -- int 0 to 100
                    ROW_NUMBER() OVER (PARTITION BY inning_number, order_number ORDER BY batter_seq_id ASC) as round
                FROM read_parquet('azure://{CONTAINER_NAME}/{pa_info_blob_name}', union_by_name=True)
                ORDER BY batter_seq_id
            """
            batter_df = con.execute(query).fetchdf()
            
            if batter_df.empty:
                print("no valid at bat data found")
                # Return empty scorebook entries instead of an error
                return {
                    "team_id": team_id,
                    "game_id": game_id,
                    "inning_number": inning_number,
                    "team_choice": team_choice,
                    "my_team_ha": my_team_ha,
                    "lineup_available": lineup_available,
                    "lineup_entries": lineup_entries,
                    "scorebook_entries": []
                }
                                        
        except Exception as e:
            # Log the error but don't raise an exception
            logger.warning(f"No plate appearance parquets found: {str(e)}")


        # Add all other fields that exist in the data
        scorebook_entries = []
        
        if not batter_df.empty:
            # Convert DataFrame to dictionary records for easier processing
            for _, row in batter_df.iterrows():
                entry = {}
                
                # Process each field in the row
                for field in batter_df.columns:
                    value = row[field]
                    # Special handling for array fields
                    if field in ARRAY_FIELDS:
                        # Convert to list if it's not already
                        if isinstance(value, (list, pd.Series, np.ndarray)):
                            if hasattr(value, 'tolist'):
                                entry[field] = value.tolist()
                            else:
                                entry[field] = list(value)
                        elif isinstance(value, str):
                            # Try to parse as JSON if it's a string
                            try:
                                entry[field] = json.loads(value)
                            except Exception as e:
                                # If parsing fails, convert the string to a single-item list
                                if value and not pd.isna(value):  # Check if value exists and is not NaN
                                    entry[field] = [value]
                                else:
                                    entry[field] = []
                                logger.warning(f"Converted string value '{value}' to list for {field}")
                        else:
                            # Default to empty list for array fields
                            entry[field] = []
                            logger.warning(f"Using empty list for {field}, original type: {type(value)}")
                    else:
                        # Handle scalar values - FIXED THIS PART
                        if isinstance(value, (np.ndarray, pd.Series, list)):
                            # Convert arrays to strings in a safe way
                            try:
                                if hasattr(value, 'tolist'):
                                    entry[field] = str(value.tolist())
                                else:
                                    entry[field] = str(list(value))
                            except:
                                entry[field] = str(value)
                        elif pd.api.types.is_numeric_dtype(type(value)):
                            entry[field] = int(value) if pd.notna(value) else 0
                        else:
                            # Use pd.isna instead of pd.notna to avoid array truth value issues
                            entry[field] = str(value) if not pd.isna(value) else ""
                
                scorebook_entries.append(entry)
        
        # Build the response
        response = {
            "team_id": team_id,
            "game_id": game_id,
            "inning_number": inning_number,
            "team_choice": team_choice,
            "my_team_ha": my_team_ha,
            "lineup_available": lineup_available,
            "lineup_entries": lineup_entries,
            "scorebook_entries": scorebook_entries
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting inning scorebook: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Detailed error: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting inning scorebook: {str(e)}"
        )
    
@router.post("/api/plate-appearance", status_code=201)
async def save_plate_appearance(data: dict = Body(...)):
    try:
        import json  # Make sure json is imported at the top if it's not already
        print("Full incoming JSON data:")
        print(json.dumps(data, indent=2))

        # Ensure required fields are present
        required_fields = ["team_id", "game_id", "inning_number", "home_or_away", "batter_seq_id"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required field: {field}"
                )
       
        # Ensure list fields are properly formatted
        for field in ARRAY_FIELDS:
            if field in data:
                if not isinstance(data[field], list):
                    try:
                        data[field] = list(data[field])
                    except:
                        data[field] = []
                
                # Log the list field for debugging
                logger.info(f"Saving {field} as: {data[field]} (type: {type(data[field])})")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        team_data_key = "home" if data["home_or_away"] == "home" else "away"
        blob_name = f"games/team_{data['team_id']}/game_{data['game_id']}/inning_{data['inning_number']}/{team_data_key}_{data['batter_seq_id']}.parquet"
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(parquet_buffer, overwrite=True)
        return {
            "status": "success",
            "message": f"Plate appearance data saved for team {data['team_id']}, game {data['game_id']}, inning {data['inning_number']}, batter sequence {data['batter_seq_id']}",
            "blob_path": blob_name
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error saving plate appearance data: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving plate appearance data: {str(e)}"
        )

@router.delete("/api/plate-appearance/{team_id}/{game_id}/{inning_number}/{team_choice}/{batter_seq_id}")
async def delete_plate_appearance(
    team_id: str,
    game_id: str, 
    inning_number: int,
    team_choice: str,
    batter_seq_id: int
):
    """
    Delete a plate appearance file using path parameters
    Parameters:
    - team_id: The team ID    - game_id: The game ID    - inning_number: The inning number    - team_choice: 'home' or 'away'    - batter_seq_id: The batter sequence ID
    """
    try:
        # Validate team_choice
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        # Create the blob path
        blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_{batter_seq_id}.parquet"
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Check if the blob exists
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_exists(blob_client):
            raise HTTPException(
                status_code=404,
                detail=f"Plate appearance not found: {blob_name}"
            )
        # Delete the blob
        blob_client.delete_blob()
        return {
            "status": "success",
            "message": f"Plate appearance deleted: {blob_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting plate appearance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting plate appearance: {str(e)}"
        )

@router.post("/{team_id}/{game_id}/{inning_number}/{team_choice}/calculate-score")
async def calculate_score(team_id: str, game_id: str, inning_number: int, team_choice: str):
    try:
        # Validate team_choice
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        # step 1: connect to parquet's
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        plate_appearance_blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_*.parquet"

        #step 2: do the aggregations of runs, outs, strikeouts, walks, hits, errors
        # Use CTE with grouping by team metadata
        batters_df = con.execute(f"""
            WITH stats AS (
                SELECT 
                    team_id,
                    my_team_ha, 
                    game_id,
                    inning_number,
                    home_or_away,
                    -- out_at,
                    -- balls_before_play,
                    -- strikes_before_play,
                    -- pitch_count,
                    -- strikes_unsure,
                    -- br_result,
                    -- wild_pitch,
                    -- passed_ball,
                    -- strikes_watching,
                    -- strikes_swinging,
                    -- ball_swinging,
                    -- fouls,
                    -- fouls_after_two_strikes,
                    -- pa_result,
                    -- hit_to,
                    -- pa_why,
                    -- pa_error_on,
                    -- br_stolen_bases,
                    -- br_error_on,
                    -- base_running_hit_around,
                    -- base_running_other,
                    -- stolen_bases,
                    -- hit_around_bases,
                    CASE WHEN br_result = 4 or pa_result = 4 THEN 1 ELSE 0 END as runs,
                    out as outs,
                    CASE WHEN pa_why IN ('KK','K') THEN 1 ELSE 0 END as strikeouts,
                    CASE WHEN pa_why IN ('BB','HBP') THEN 1 ELSE 0 END as walks,
                    CASE WHEN pa_why IN ('H','HH','HR','GS','S','B','GS') THEN 1 ELSE 0 END as hits,
                    CASE WHEN array_length(br_error_on) > 0 AND array_length(pa_error_on) > 0 THEN 2
                        WHEN array_length(pa_error_on) > 0 THEN 1
                        WHEN array_length(br_error_on) > 0 THEN 1
                        WHEN pa_result IN ('E') THEN 1
                        ELSE 0 END as errors
                FROM read_parquet('azure://{CONTAINER_NAME}/{plate_appearance_blob_name}',union_by_name=True)
            )
            SELECT 
                team_id,
                my_team_ha,
                game_id,
                inning_number,
                home_or_away,
                SUM(runs) as runs,
                SUM(outs) as outs,
                SUM(strikeouts) as strikeouts,
                SUM(walks) as walks,
                SUM(hits) as hits,
                SUM(errors) as errors
            FROM stats
            GROUP BY team_id, my_team_ha, game_id, inning_number, home_or_away
        """).fetchdf()
        if batters_df.empty:
                print("no valid at bat data found")
        #step : I want to save the response to a parquet
        blob_name = f"games/team_{team_id}/game_{game_id}/box_score/{team_choice}_{inning_number}.parquet"
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        parquet_buffer = BytesIO()
        batters_df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        blob_client.upload_blob(parquet_buffer, overwrite=True)

    except Exception as e:
        logger.error(f"Error calculating score: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating inning score: {str(e)}"
        )
@router.get("/{team_id}/{game_id}/summary")
async def get_game_summary(team_id: str, game_id: str):
    """
    Get a summary of all innings for a game with box score format
    """
    try:
        con = get_duckdb_connection()
        
        # Step 1: Get game header info first (this should always exist)
        game_header_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        try:
            game_header_df = con.execute(f"""
                SELECT 
                    user_team,
                    coach,
                    away_team_name as opponent_name,
                    event_date,
                    event_hour,
                    event_minute,
                    field_name,
                    field_location,
                    field_type,
                    field_temperature,
                    game_status,
                    my_team_ha,
                    game_id
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_header_blob_name}')
            """).fetchdf()
            
            if game_header_df.empty:
                logger.warning(f"Game header not found for team {team_id}, game {game_id}")
                # Create default header
                game_header = {
                    "user_team": "Unknown",
                    "coach": "Unknown",
                    "opponent_name": "Unknown",
                    "event_date": "",
                    "game_id": game_id,
                    "my_team_ha": "home"  # Default to home
                }
            else:
                game_header = game_header_df.to_dict(orient='records')[0]
                
        except Exception as e:
            logger.warning(f"Error reading game header: {str(e)}")
            # Create default header
            game_header = {
                "user_team": "Unknown",
                "coach": "Unknown",
                "opponent_name": "Unknown",
                "event_date": "",
                "game_id": game_id,
                "my_team_ha": "home"  # Default to home
            }
        
        # Step 2: Get box score data
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}/box_score/*.parquet"
        
        try:
            game_info_df = con.execute(f"""
                SELECT 
                    team_id,
                    game_id,
                    inning_number,
                    home_or_away,
                    runs,
                    outs,
                    strikeouts,
                    walks,
                    hits,
                    errors
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}', union_by_name=True)
            """).fetchdf()
        except Exception as e:
            logger.warning(f"Error reading box score data: {str(e)}")
            game_info_df = pd.DataFrame()
        
        # Step 3: Create a complete inning-by-inning structure with zeros for missing data
        # Initialize data structure for all innings (1-7) for both home and away
        innings_data = {}
        for inning in range(1, 8):  # Assuming 7 innings max
            innings_data[str(inning)] = {
                "home": {
                    "runs": 0,
                    "hits": 0,
                    "errors": 0,
                    "walks": 0,
                    "outs": 0,
                    "strikeouts": 0
                },
                "away": {
                    "runs": 0,
                    "hits": 0,
                    "errors": 0,
                    "walks": 0,
                    "outs": 0,
                    "strikeouts": 0
                }
            }
        
        # Initialize totals
        totals = {
            "home": {
                "runs": 0,
                "hits": 0,
                "errors": 0,
                "walks": 0,
                "outs": 0,
                "strikeouts": 0
            },
            "away": {
                "runs": 0,
                "hits": 0,
                "errors": 0,
                "walks": 0,
                "outs": 0,
                "strikeouts": 0
            }
        }
        
        # Fill in data from the DataFrame if it exists
        if not game_info_df.empty:
            for _, row in game_info_df.iterrows():
                inning_number = str(row['inning_number'])
                team_choice = row['home_or_away'].lower()
                # Determine the opposing team (for errors)
                opposing_team = "away" if team_choice == "home" else "home"
                
                # Skip if inning number is out of range
                if inning_number not in innings_data:
                    continue
                
                # Update inning data for runs, hits, walks, outs, strikeouts
                innings_data[inning_number][team_choice].update({
                    "runs": int(row['runs']) if pd.notna(row['runs']) else 0,
                    "hits": int(row['hits']) if pd.notna(row['hits']) else 0,
                    "walks": int(row['walks']) if pd.notna(row['walks']) else 0,
                    "outs": int(row['outs']) if pd.notna(row['outs']) else 0,
                    "strikeouts": int(row['strikeouts']) if pd.notna(row['strikeouts']) else 0
                })
                
                # Errors are attributed to the OPPOSING team (fielding team)
                innings_data[inning_number][opposing_team]["errors"] += int(row['errors']) if pd.notna(row['errors']) else 0
                
                # Update totals for runs, hits, walks, outs, strikeouts
                totals[team_choice].update({
                    "runs": totals[team_choice]["runs"] + (int(row['runs']) if pd.notna(row['runs']) else 0),
                    "hits": totals[team_choice]["hits"] + (int(row['hits']) if pd.notna(row['hits']) else 0),
                    "walks": totals[team_choice]["walks"] + (int(row['walks']) if pd.notna(row['walks']) else 0),
                    "outs": totals[team_choice]["outs"] + (int(row['outs']) if pd.notna(row['outs']) else 0),
                    "strikeouts": totals[team_choice]["strikeouts"] + (int(row['strikeouts']) if pd.notna(row['strikeouts']) else 0)
                })
                
                # Update errors for the OPPOSING team
                totals[opposing_team]["errors"] += int(row['errors']) if pd.notna(row['errors']) else 0
        
        # Build the final response
        response = {
            "game_header": game_header,
            "innings": innings_data,
            "totals": totals
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting game summary: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Detailed error: {error_details}")
        
        # Even in case of error, return a valid response structure with zeros
        # This ensures the frontend always gets a valid response format
        default_response = {
            "game_header": {
                "user_team": "Error",
                "coach": "",
                "opponent_name": "",
                "event_date": "",
                "game_id": game_id,
                "my_team_ha": "home"
            },
            "innings": {str(i): {"home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0}, 
                                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0}} 
                        for i in range(1, 8)},
            "totals": {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0}
            },
            "error": str(e)
        }
        
        return default_response


