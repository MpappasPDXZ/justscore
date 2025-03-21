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
    slap: int  # 1 for yes, 0 for no
    late_swings: int  # 1 for yes, 0 for no
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
    pitch_count: int
    balls_before_play: int
    strikes_before_play:  Optional[int] = 0
    strikes_unsure:  Optional[int] = 0
    strikes_watching:  Optional[int] = 0
    strikes_swinging:  Optional[int] = 0
    ball_swinging:  Optional[int] = 0
    fouls:  Optional[int] = 0
    wild_pitches: Optional[int] = 0
    wild_pitch:  Optional[int] = 0
    passed_ball:  Optional[int] = 0
    rbi:  Optional[int] = 0
    late_swings: Optional[int] = 0
    qab: Optional[int] = 0
    hard_hit: Optional[int] = 0
    slap: Optional[int] = 0
    sac: Optional[int] = 0
    pa_result: str
    pa_why: Optional[str] = ""
    hit_to:  Optional[int] = 0
    pa_error_on:  Optional[str] = 0
    br_result: Optional[int] = 0
    br_stolen_bases: Optional[str] = "" #this is a list of text wtih values "1","2","3","4" how do I parse this?
    br_error_on: Optional[str] = ""
    base_running_hit_around:  Optional[str] = ""
###########################################################
# 1 - Plate appearance
###########################################################
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
        # Properly handle array fields during save
        ARRAY_FIELDS = ["base_running_hit_around", "br_stolen_bases", "pa_error_on", "br_error_on"]
        for field in ARRAY_FIELDS:
            if field in data:
                # If it's already a list, keep it
                if isinstance(data[field], list):
                    continue
                # If it's a string representation of a list, parse it
                elif isinstance(data[field], str):
                    try:
                        if data[field].startswith('[') and data[field].endswith(']'):
                            data[field] = ast.literal_eval(data[field])
                        elif data[field]:  # If not empty string
                            data[field] = [data[field]]
                        else:  # Empty string
                            data[field] = []
                    except:
                        data[field] = [] if not data[field] else [data[field]]
                # If it's some other value, wrap it in a list if not empty
                elif data[field]:
                    data[field] = [data[field]]
                else:
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
        print("------1 - SAVE PLATE APPEARANCE ------------")
        print(f": {list(df.columns)}")
        print(f"first row sample: {df.iloc[0].to_dict() if not df.empty else 'No data'}")
        print("----------------------------------------------")
        return {
            "status": "success",
            "message": f"Plate appearance data saved for team {data['team_id']}, game {data['game_id']}, inning {data['inning_number']}, batter sequence {data['batter_seq_id']}",
            "blob_path": blob_name,
            "team_choice": data["home_or_away"],
            "my_team_ha": data.get("my_team_ha", "home")
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
###########################################################
# 2 - calculate the inning score
###########################################################
@router.post("/{team_id}/{game_id}/{inning_number}/{team_choice}/calculate-score")
async def calculate_score(team_id: str, game_id: str, inning_number: int, team_choice: str):
    try:
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        plate_appearance_blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_*.parquet"    
        try:
            schema_query = f"""
                DESCRIBE SELECT * FROM read_parquet('azure://{CONTAINER_NAME}/{plate_appearance_blob_name}', union_by_name=True)
            """
            schema_df = con.execute(schema_query).fetchdf()
            # Check for required columns
            required_columns = [
                "team_id", "game_id", "inning_number", "home_or_away", 
                "pa_result", "pa_why", "br_result", "pa_error_on", "br_error_on",
                "hard_hit", "strikes_before_play", "pitch_count", "out"
            ]
            
            available_columns = set(schema_df['column_name'].tolist())
            missing_columns = [col for col in required_columns if col not in available_columns]
            if missing_columns:
                return {
                    "status": "error",
                    "message": f"Missing required columns: {missing_columns}",
                    "detail": "The parquet files for this inning are missing required columns. Check your data collection process."
                }
        except Exception as e:
            error_msg = str(e)
            if "Failed to read Parquet file" in error_msg:
                return {
                    "status": "error",
                    "message": f"No plate appearance data found for {team_choice} team in inning {inning_number}",
                    "detail": "Please record plate appearances before calculating scores."
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error checking parquet schema: {error_msg}",
                    "detail": "There was an error accessing or analyzing the parquet files."
                }
        
        # Continue with your existing logic for valid data
        batters_df = pd.DataFrame()
        try:
            simple_query = f"""
                SELECT  *
                FROM read_parquet('azure://{CONTAINER_NAME}/{plate_appearance_blob_name}', union_by_name=True)
            """
            df = con.execute(simple_query).fetchdf()
            if not df.empty:
                # Calculate strikes - total strikes and total pitches
                df['total_strikes'] = df.apply(lambda row: 
                    1 if row['pa_why'] in ['H', 'B','HR','GS','E','C','K','KK','GO','FO','LO','FB'] else 0, axis=1) + df['strikes_before_play']
                df['total_pitches'] = df['pitch_count']
                # Calculate on-base metrics
                df['on_base_count'] = df.apply(lambda row: 1 if str(row['pa_result']) in ['1', '2', '3', '4'] else 0, axis=1)
                df['plate_appearances'] = 1  # Each row is one plate appearance
                # Calculate basic stats
                df['out'] = df.apply(lambda row: 1 if row['out'] == 1 else 0, axis=1)
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
                aggregated = df.groupby(['team_id','game_id', 'inning_number', 'home_or_away']).agg({
                    'runs': 'sum',
                    'hits': 'sum',
                    'errors': 'sum',
                    'strikeouts': 'sum',
                    'walks': 'sum',
                    'out': 'sum',
                    'hard_hit': 'sum',
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
                numeric_cols = ['runs', 'hits', 'errors', 'strikeouts', 'walks', 'out', 'hard_hit', 'strike_percent', 'on_base_percent', 'on_first_base', 'on_second_base', 'on_third_base']
                for col in numeric_cols:
                    if col in ['strike_percent', 'on_base_percent']:
                        batters_df[col] = batters_df[col].astype(int)                 
                
                # After all calculations, verify the batters_df is not empty
                if batters_df.empty:
                    print("ERROR: batters_df is empty after aggregation!")
                    return {
                        "status": "error",
                        "message": "Failed to calculate score: The aggregated data is empty",
                        "detail": "Check your original data and aggregation logic"
                    }
                
                # Print diagnostic info about the DataFrame
                print("------2 - CALCULATE-INNING-SCORE ------------")
                print(f"calculate inning score batters_dfcolumns: {list(batters_df.columns)}")
                print(f"First row sample: {batters_df.iloc[0].to_dict() if not batters_df.empty else 'No data'}")
                print("----------------------------------------------")
            else:
                print("ERROR: Source DataFrame is empty - no data in parquet files")
                return {
                    "status": "error",
                    "message": f"No plate appearance data found for {team_choice} team in inning {inning_number}",
                    "detail": "The parquet files contain no data. Please record plate appearances before calculating scores."
                }
        except Exception as e:
            # Print specific error
            print(f"ERROR during data processing: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Error processing data: {str(e)}",
                "detail": "There was an error during data calculation."
            }
            
        # Save the response to a parquet
        blob_name = f"games/team_{team_id}/game_{game_id}/box_score/{team_choice}_{inning_number}.parquet"
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        
        # Now save to Azure
        try:
            parquet_buffer = BytesIO()
            batters_df.to_parquet(parquet_buffer)            
            parquet_buffer.seek(0)
            blob_client.upload_blob(parquet_buffer, overwrite=True)            
            return {
                "status": "success",
                "message": f"Score calculated for team {team_id}, game {game_id}, inning {inning_number}",
                "data": batters_df.to_dict(orient='records'),
                "columns": list(batters_df.columns),
                "row_count": len(batters_df)
            }
        except Exception as e:
            logger.error(f"Error saving parquet to Azure: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save parquet file: {str(e)}",
                "detail": "There was an error saving the calculated data."
            }

    except Exception as e:
        logger.error(f"Error calculating score: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating inning score: {str(e)}"
        )
###########################################################
# 3 - Read the box score from calc score
###########################################################
@router.get("/{team_id}/{game_id}/summary")
async def get_game_summary(team_id: str, game_id: str):
    """
    Get a summary of all innings for a game with box score format
    """
    try:
        con = get_duckdb_connection()
        # Step 1: Get game header info
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"       
        # Initialize game_header with default values
        game_header = {
            "user_team": "Unknown", 
            "coach": "Unknown", 
            "opponent_name": "Unknown",
            "event_date": "", 
            "event_hour": 0,
            "event_minute": 0,
            "field_name": "",
            "field_location": "",
            "field_type": "",
            "field_temperature": 0,
            "game_status": "In Progress",
            "my_team_ha": "home",
            "game_id": game_id
        }
        
        # Try to get header info but continue if not found
        try:
            header_query = f"""
                SELECT 
                  user_team
                , my_team_ha
                , coach
                , away_team_name as opponent_name
                , event_date
                , event_hour
                , event_minute
                , field_name
                , field_location
                , field_type
                , field_temperature
                , game_status
                , game_id
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}')
            """
            game_header_df = con.execute(header_query).fetchdf()
            if not game_header_df.empty:
                print("------3a - GAME-HEADER-INFO ------------")
                print(f"game_header_df columns: {list(game_header_df.columns)}")
                print(f"game_header_df sample: {game_header_df.iloc[0].to_dict() if not game_header_df.empty else 'No data'}")
                print("----------------------------------------------")
                game_header = game_header_df.iloc[0].to_dict()
                team_choice = game_header['my_team_ha'].lower()
        except Exception as e:
            logger.warning(f"Error getting game header: {str(e)}")
            # Continue with default header values
        
        # Step 2: Initialize data structures
        innings_data = {}
        for i in range(1, 8):
            innings_data[str(i)] = {
                "home": {
                    "runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                    "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0, 
                    "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []
                },
                "away": {
                    "runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                    "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0,
                    "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []
                }
            }
        totals = {
            "home": {
                "runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0,
                "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []
            },
            "away": {
                "runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, 
                "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0,
                "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []
            }
        }
        
        # Step 3: Get box score data for both teams
        all_innings_data = {}
        
        # Get all innings data from both teams
        try:
            game_innings_blob_name = f"games/team_{team_id}/game_{game_id}/box_score/*.parquet"              
            query = f"""
                SELECT *   
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_innings_blob_name}', union_by_name=True)
                """
            innings_df = con.execute(query).fetchdf()
            print("------3b - INNINGS-DF-RETRIEVAL ------------")
            print(f"innings_df columns: {list(innings_df.columns)}")
            print(f"innings_df sample: {innings_df.iloc[0].to_dict() if not innings_df.empty else 'No data'}")
            print("----------------------------------------------")
            
            if not innings_df.empty:                        
                for _, row in innings_df.iterrows():
                    inning = int(row['inning_number'])
                    team = row['home_or_away'].lower()
                    
                    if 1 <= inning <= 7:  # Only process innings 1-7
                        if inning not in all_innings_data:
                            all_innings_data[inning] = {"home": {}, "away": {}}
                        
                        all_innings_data[inning][team] = {
                            "runs": int(row['runs']),
                            "hits": int(row['hits']),
                            "errors": int(row['errors']),
                            "walks": int(row['walks']),
                            "outs": int(row['out']),
                            "strikeouts": int(row['strikeouts']),
                            "hard_hit": int(row['hard_hit']),
                            "strike_percent": int(row['strike_percent']),
                            "on_base_percent": int(row['on_base_percent']),
                            "on_first_base": int(row['on_first_base']),
                            "on_second_base": int(row['on_second_base']),
                            "on_third_base": int(row['on_third_base']),
                            "runners_on_base": []
                        }
        except Exception as e:
            error_msg = str(e)
            if "No files found that match the pattern" in error_msg:
                logger.warning(f"No box score files found: {error_msg}")
            else:
                logger.error(f"Error retrieving box score data: {error_msg}")
        
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
                    if "errors" in inning_data[team]:
                        innings_data[inning_str][opposite]["errors"] = inning_data[team]["errors"]
                        swapped_totals[opposite]["errors"] += inning_data[team]["errors"]
                    
                    # Swap strike_percent - pitcher's strike percentage
                    if "strike_percent" in inning_data[team]:
                        innings_data[inning_str][opposite]["strike_percent"] = inning_data[team]["strike_percent"]
                        # Store non-zero values for averaging later
                        if inning_data[team]["strike_percent"] > 0:
                            swapped_totals[opposite]["strike_percent"].append(inning_data[team]["strike_percent"])
        
        # Step 5: Calculate totals
        for team in ["home", "away"]:
            team_innings = [innings_data[str(i)][team] for i in range(1, 8) if str(i) in innings_data]
            if team_innings:
                # Sum up the numeric stats (excluding errors and strike_percent which we've already swapped)
                for stat in ["runs", "hits", "walks", "outs", "strikeouts", "hard_hit", "on_first_base", "on_second_base", "on_third_base"]:
                    # Check if all innings have this stat before summing
                    totals[team][stat] = sum(inning.get(stat, 0) for inning in team_innings)
                
                # For on_base_percent, take the average of non-zero values
                on_base_values = [inning["on_base_percent"] for inning in team_innings 
                                  if "on_base_percent" in inning and inning["on_base_percent"] > 0]
                if on_base_values:
                    totals[team]["on_base_percent"] = sum(on_base_values) / len(on_base_values)
            
            # Set the errors from our swapped totals
            totals[team]["errors"] = swapped_totals[team]["errors"]
            
            # Set strike_percent from our swapped totals, using average
            strike_values = swapped_totals[team]["strike_percent"]
            totals[team]["strike_percent"] = sum(strike_values) / len(strike_values) if strike_values else 0
        
        # Return the formatted response that matches the required JSON structure
        response = {
            "game_header": game_header,
            "innings": innings_data,
            "totals": totals
        }           
        return response
        
    except Exception as e:
        logger.error(f"Error in get_game_summary: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a valid response structure with error info
        return {
            "game_header": {
                "user_team": "Error", 
                "coach": "Unknown", 
                "opponent_name": "Unknown",
                "event_date": "", 
                "event_hour": 0,
                "event_minute": 0,
                "field_name": "",
                "field_location": "",
                "field_type": "",
                "field_temperature": 0,
                "game_status": "Error",
                "my_team_ha": "home",
                "game_id": game_id
            },
            "innings": {str(i): {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []}
            } for i in range(1, 8)},
            "totals": {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "outs": 0, "strikeouts": 0, "hard_hit": 0, "strike_percent": 0, "on_base_percent": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0, "runners_on_base": []}
            }
        }

###########################################################
# 4 - get the inning scorebook by attempt (this is the big data grid)
###########################################################
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
        #--------------------------------
        # Step 3: Get at bat data from the new file structure
        #--------------------------------
        pa_info_blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_choice}_*.parquet"
        batter_df = pd.DataFrame()  # Initialize empty DataFrame
        try:
            query = f"""
                SELECT * 
                FROM read_parquet('azure://{CONTAINER_NAME}/{pa_info_blob_name}', union_by_name=True)
                ORDER BY batter_seq_id
            """
            batter_df = con.execute(query).fetchdf()
        except Exception as e:
            # Log the error but don't raise an exception
            logger.warning(f"Reading plate appearance info: {str(e)}")
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
                
        # Process array fields - convert string representations to actual arrays
        array_fields = ["base_running_hit_around", "br_error_on", "br_stolen_bases", "pa_error_on"]
        for field in array_fields:
            if field in batter_df.columns:
                def convert_to_array(val):
                    try:
                        # Handle numpy arrays
                        if isinstance(val, np.ndarray):
                            return val.tolist()
                        
                        # Handle None/NaN values
                        if pd.isna(val) or val is None:
                            return []
                            
                        # Handle empty strings
                        if isinstance(val, str) and not val:
                            return []
                            
                        # Handle string representations of lists
                        if isinstance(val, str):
                            if val.startswith('[') and val.endswith(']'):
                                try:
                                    result = ast.literal_eval(val)
                                    return result if isinstance(result, list) else [result]
                                except (SyntaxError, ValueError):
                                    # If literal_eval fails, try a direct split approach
                                    if ',' in val:
                                        return [item.strip() for item in val.strip('[]').split(',')]
                                    # For single item in brackets
                                    return [val.strip('[]')]
                            else:
                                # Single string value, not in brackets
                                return [val]
                        
                        # Handle existing lists
                        if isinstance(val, list):
                            return val
                        
                        # Handle other scalar values (numbers)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            # Convert numbers to strings for consistency
                            return [str(int(val))]
                            
                        # Fallback - any other type gets converted to string
                        return [str(val)]
                    
                    except Exception as e:
                        # Silent error handling - just return empty array
                        return []
                
                # Apply the conversion function to the column
                batter_df[field] = batter_df[field].apply(convert_to_array)
        
        # Additional processing for hit_to - ensure it's a string
        if 'hit_to' in batter_df.columns:
            batter_df['hit_to'] = batter_df['hit_to'].apply(
                lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else 
                         (str(x) if isinstance(x, str) and x else "")
            )
        
        # Ensure all expected fields are present, add with default values if missing
        expected_fields = {
            "ball_swinging": 0, "balls_before_play": 0, "base_running_hit_around": [], 
            "batter_seq_id": 0, "batting_order_position": 0, "br_error_on": [], 
            "br_result": 0, "br_stolen_bases": [], "fouls": 0, "game_id": "", 
            "gameId": "", "hard_hit": 0, "hit_to": "", "home_or_away": "", 
            "inning_number": 0, "late_swings": 0, "my_team_ha": "", 
            "order_number": 0, "out": 0, "out_at": 0, "pa_error_on": [], 
            "pa_result": "", "pa_why": "", "passed_ball": 0, "pitch_count": 0, 
            "qab": 0, "rbi": 0, "sac": 0, "slap": 0, "strikes_before_play": 0, 
            "strikes_swinging": 0, "strikes_unsure": 0, "strikes_watching": 0, 
            "team_id": "", "teamId": "", "wild_pitch": 0
        }
        
        for field, default_value in expected_fields.items():
            if field not in batter_df.columns:
                batter_df[field] = default_value
                
        # Convert DataFrame to records but ensure array fields are properly handled
        records = []
        for _, row in batter_df.iterrows():
            record = {}
            for column in batter_df.columns:
                value = row[column]
                # Ensure arrays are properly converted
                if column in array_fields:
                    record[column] = value if isinstance(value, list) else []
                else:
                    record[column] = value
            
            # Special handling: If pa_why is "E" (error) but pa_error_on is empty, add a default value
            if record.get('pa_why') == 'E' and (not record.get('pa_error_on') or len(record.get('pa_error_on', [])) == 0):
                hit_to = record.get('hit_to', '')
                if hit_to and (str(hit_to).isdigit() or (isinstance(hit_to, (int, float)) and not pd.isna(hit_to))):
                    # Make sure it's a string
                    record['pa_error_on'] = [str(hit_to) if isinstance(hit_to, str) else str(int(hit_to))]
                else:
                    # Just use a default position if hit_to isn't available
                    record['pa_error_on'] = ['6']  # Default to shortstop
            
            records.append(record)
            
        scorebook_entries = records
        
        print("------3b - scorebook-DF-RETRIEVAL ------------")
        print(f"batter_df columns: {list(batter_df.columns)}")
        print(f"batter_df sample: {batter_df.iloc[0].to_dict() if not batter_df.empty else 'No data'}")
        print("----------------------------------------------")
        
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
        
        # Final check - make sure all array fields are properly formatted (without logging)
        for entry in response["scorebook_entries"]:
            for field in array_fields:
                if field in entry and not isinstance(entry[field], list):
                    entry[field] = [] if entry[field] is None else [entry[field]]
        
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