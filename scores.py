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
import logging
import ast

# Disable the verbose Azure Storage logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)

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

def count_errors(error_array):
    """Count elements in a NumPy array"""
    if error_array is None:
        return 0
        
    try:
        # For NumPy arrays
        if isinstance(error_array, np.ndarray):
            # If it's an empty array
            if error_array.size == 0:
                return 0
            # Count number of elements in array
            return len(error_array)
        # For any other type, return 0
        return 0
    except Exception as e:
        print(f"Error counting: {str(e)}")
        return 0

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
                        else:
                            # Default to empty list for array fields
                            entry[field] = []
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
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        plate_appearance_blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_*.parquet"
        # Get the data using a very simple query
        try:
            simple_query = f"""
                SELECT 
                    team_id,
                    my_team_ha, 
                    game_id,
                    inning_number,
                    home_or_away,
                    br_result,
                    pa_result,
                    out,
                    pa_why,
                    pa_error_on,
                    br_error_on,
                    hard_hit,
                    strikes_before_play,
                    pitch_count
                FROM read_parquet('azure://{CONTAINER_NAME}/{plate_appearance_blob_name}', union_by_name=True)
            """
            df = con.execute(simple_query).fetchdf()
            if not df.empty:
                # Convert hard_hit string "HH" to numeric 1, everything else to 0
                df['hard_hits'] = df.apply(lambda row: 1 if str(row['hard_hit']).upper() == 'HH' else 0, axis=1)
                # Calculate strikes - total strikes and total pitches
                df['total_strikes'] = df.apply(lambda row: 
                    1 if row['pa_why'] in ['H', 'B','HR','GS','E','C','K','KK','GO','FO','LO','FB'] else 0, axis=1) + df['strikes_before_play']
                df['total_pitches'] = df['pitch_count']
                # Calculate on-base metrics
                df['on_base_count'] = df.apply(lambda row: 1 if str(row['pa_result']) in ['1', '2', '3', '4'] else 0, axis=1)
                df['plate_appearances'] = 1  # Each row is one plate appearance
                # Calculate basic stats
                df['runs'] = df.apply(lambda row: 1 if row['br_result'] == 4 or str(row['pa_result']) == '4' else 0, axis=1)
                df['strikeouts'] = df.apply(lambda row: 1 if row['pa_why'] in ['KK', 'K'] else 0, axis=1)
                df['walks'] = df.apply(lambda row: 1 if row['pa_why'] in ['BB', 'HBP'] else 0, axis=1)
                df['hits'] = df.apply(lambda row: 1 if row['pa_why'] in ['H', 'HR', 'GS', 'S', 'B'] else 0, axis=1)
                #I want to convert this to a string and then remove brackets, commas, spaces, [, ] and then count the number of characters
                df['errors'] = df.apply(
                    lambda row: 
                        # Count elements in error arrays
                        count_errors(row['pa_error_on']) + count_errors(row['br_error_on']) +
                        # If pa_why is 'E' AND there are no elements in the error arrays, add 1
                        (1 if row['pa_why'] == 'E' and 
                             count_errors(row['pa_error_on']) == 0 and 
                             count_errors(row['br_error_on']) == 0 
                         else 0), 
                    axis=1
                )
                df['on_first_base']  = df.apply(lambda row: 1 if (row['pa_result'] == 1 or row['br_result'] == 1) and not (row['br_result'] in [2,3,4] or row['out'] == 1 or row['pa_result'] in [0,2,3,4]) else 0, axis=1)
                df['on_second_base'] = df.apply(lambda row: 1 if (row['pa_result'] == 2 or row['br_result'] == 2) and not (row['br_result'] in [3,4] or row['out'] == 1 or row['pa_result'] in [0,3,4]) else 0, axis=1)
                df['on_third_base']  = df.apply(lambda row: 1 if (row['pa_result'] == 3 or row['br_result'] == 3) and not (row['br_result'] in [4] or row['out'] == 1 or row['pa_result'] in [0,4]) else 0, axis=1)
                # Group by team info and sum the basic counts
                print('--------------------------------')
                print(df.head(10))
                print('--------------------------------')
                aggregated = df.groupby(['team_id','game_id', 'inning_number', 'home_or_away']).agg({
                    'runs': 'sum',
                    'hits': 'sum',
                    'errors': 'sum',
                    'strikeouts': 'sum',
                    'walks': 'sum',
                    'out': 'sum',
                    'hard_hits': 'sum',
                    'total_strikes': 'sum',
                    'total_pitches': 'sum',
                    'on_base_count': 'sum',
                    'plate_appearances': 'sum',
                    'on_first_base': 'sum',
                    'on_second_base': 'sum',
                    'on_third_base': 'sum'  
                }).reset_index()
                # Calculate percentages - ensure they're stored as ints
                aggregated['strike_percent'] = (aggregated['total_strikes'] / aggregated['total_pitches']*100 ).fillna(0.0).astype(int)
                aggregated['on_base_percent'] = (aggregated['on_base_count'] / aggregated['plate_appearances']*100).fillna(0.0).astype(int)  
                # Explicitly round to 2 decimal places
                aggregated['strike_percent'] = aggregated['strike_percent'].round(0)
                aggregated['on_base_percent'] = aggregated['on_base_percent'].round(0)
                # Drop the working columns
                aggregated = aggregated.drop(['total_strikes', 'total_pitches', 'on_base_count', 'plate_appearances'], axis=1)
                batters_df = aggregated               
                # Make absolutely sure the column data types are preserved
                numeric_cols = ['runs', 'hits', 'errors', 'strikeouts', 'walks', 'out', 'hard_hits', 'strike_percent', 'on_base_percent', 'on_first_base', 'on_second_base', 'on_third_base']
                for col in numeric_cols:
                    if col in ['strike_percent', 'on_base_percent']:
                        batters_df[col] = batters_df[col].astype(int)                 
            else:
               pass
        except Exception as e:
            # Any error in processing, fall back to default values
               pass
            
        # Save the response to a parquet
        print(f"{team_choice} {inning_number}")
        print(batters_df.head(10))
        blob_name = f"games/team_{team_id}/game_{game_id}/box_score/{team_choice}_{inning_number}.parquet"
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        parquet_buffer = BytesIO()
        batters_df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        blob_client.upload_blob(parquet_buffer, overwrite=True)
        return {
            "status": "success",
            "message": f"Score calculated for team {team_id}, game {game_id}, inning {inning_number}",
            "data": batters_df.to_dict(orient='records')
        }

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
        
        # Step 1: Get game header info
        game_header_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        game_header = {"user_team": "Unknown", "coach": "Unknown", "opponent_name": "Unknown",
                      "event_date": "", "game_id": game_id, "my_team_ha": "home"}
        
        try:
            header_query = f"""
                SELECT 
                    user_team,
                    coach,
                    away_team_name as opponent_name,
                    event_date,
                    my_team_ha,
                    game_id
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_header_blob_name}')
            """
            game_header_df = con.execute(header_query).fetchdf()
            
            if not game_header_df.empty:
                game_header = game_header_df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Error getting game header: {str(e)}")
        
        # Step 2: Initialize data structures
        innings_data = {}
        for i in range(1, 8):
            innings_data[str(i)] = {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                        "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                        "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0}
            }
        
        totals = {
            "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                    "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0},
            "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                    "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0 }
        }
        
        # Step 3: Get box score data for each team
        all_innings_data = {}
        
        for team_choice in ["home", "away"]:
            try:
                game_info_blob_name = f"games/team_{team_id}/game_{game_id}/box_score/{team_choice}_*.parquet"
                
                try:                        
                    # Get data for this team
                    query = f"""
                    SELECT 
                        inning_number,
                        home_or_away,
                            CAST(runs AS INTEGER) as runs,
                            CAST(strikeouts AS INTEGER) as strikeouts,
                            CAST(walks AS INTEGER) as walks,
                            CAST(hits AS INTEGER) as hits,
                            CAST(errors AS INTEGER) as errors,
                            CAST(out AS INTEGER) as outs,
                            CAST(hard_hits AS INTEGER) as hard_hits,
                            CAST(strike_percent AS INTEGER) as strike_percent,
                            CAST(on_base_percent AS INTEGER) as on_base_percent,
                            CAST(on_first_base AS INTEGER) as on_first_base,
                            CAST(on_second_base AS INTEGER) as on_second_base,
                            CAST(on_third_base AS INTEGER) as on_third_base
                    FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}', union_by_name=True)
                    """                   
                    df = con.execute(query).fetchdf()
                except Exception as e:
                    logger.warning(f"Error executing query: {str(e)}")
                    logger.warning(f"Skipping {team_choice} data")
                    continue
                
                # Store the data by inning for later processing
                if not df.empty:
                    for _, row in df.iterrows():
                        inning = int(row['inning_number'])
                        if 1 <= inning <= 7:  # Only process innings 1-7
                            if inning not in all_innings_data:
                                all_innings_data[inning] = {"home": {}, "away": {}}
                            
                            # Store the data
                            team_key = team_choice.lower()
                            all_innings_data[inning][team_key] = {
                        "runs": int(row['runs']),
                        "hits": int(row['hits']),
                        "errors": int(row['errors']),
                        "walks": int(row['walks']),
                        "outs": int(row['outs']),
                        "strikeouts": int(row['strikeouts']),
                                "hard_hits": int(row['hard_hits']),
                                "strike_percent": int(row['strike_percent']),
                                "on_base_percent": int(row['on_base_percent']),
                                "on_first_base": int(row['on_first_base']),
                                "on_second_base": int(row['on_second_base']),
                                "on_third_base": int(row['on_third_base'])
                            }
            except Exception as e:
                logger.warning(f"Error getting {team_choice} data: {str(e)}")
                import traceback
                logger.warning(f"Detailed error info: {traceback.format_exc()}")
        
        # Step 4: Process the data, swapping errors and strike_percent
        swapped_totals = {
            "home": {"errors": 0, "strike_percent": []},
            "away": {"errors": 0, "strike_percent": []}
        }
        
        for inning, inning_data in all_innings_data.items():
            inning_str = str(inning)
            
            # Copy most stats directly
            for team in ["home", "away"]:
                if team in inning_data:
                    for stat, value in inning_data[team].items():
                        # Skip errors and strike_percent, we'll handle them separately
                        if stat not in ["errors", "strike_percent"]:
                            innings_data[inning_str][team][stat] = value
            
            # Now swap errors and strike_percent between teams
            for team, opposite in [("home", "away"), ("away", "home")]:
                if team in inning_data:
                    # Swap errors - defensive team's errors
                    innings_data[inning_str][opposite]["errors"] = inning_data[team]["errors"]
                    swapped_totals[opposite]["errors"] += inning_data[team]["errors"]
                    # Swap strike_percent - pitcher's strike percentage
                    innings_data[inning_str][opposite]["strike_percent"] = inning_data[team]["strike_percent"]
                    # Store non-zero values for averaging later
                    if inning_data[team]["strike_percent"] > 0:
                        swapped_totals[opposite]["strike_percent"].append(inning_data[team]["strike_percent"])
        
        # Step 5: Calculate totals
        for team in ["home", "away"]:
            team_innings = [innings_data[str(i)][team] for i in range(1, 8) if innings_data[str(i)][team]]
            if team_innings:
                # Sum up the numeric stats (excluding errors and strike_percent which we've already swapped)
                for stat in ["runs", "hits", "walks", "outs", "strikeouts", "hard_hits"]:
                    totals[team][stat] = sum(inning[stat] for inning in team_innings)
                
                # For on_base_percent, take the average of non-zero values
                on_base_values = [inning["on_base_percent"] for inning in team_innings if inning["on_base_percent"] > 0]
                totals[team]["on_base_percent"] = sum(on_base_values) / len(on_base_values) if on_base_values else 0
            
            # Set the errors from our swapped totals
            totals[team]["errors"] = swapped_totals[team]["errors"]
            
            # Set strike_percent from our swapped totals, using average
            strike_values = swapped_totals[team]["strike_percent"]
            totals[team]["strike_percent"] = sum(strike_values) / len(strike_values) if strike_values else 0
        
        # Build the response
        response = {
            "game_header": game_header,
            "innings": innings_data,
            "totals": totals
        }
        #formatted print
        return response
        
    except Exception as e:
        logger.error(f"Error in get_game_summary: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a valid response structure even on error
        default_response = {
            "game_header": {"user_team": "Error", "game_id": game_id, "my_team_ha": "home"},
            "innings": {str(i): {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0}
            } for i in range(1, 8)},
            "totals": {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hits": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0}
            },
            "error": str(e)
        }
        
        return default_response


