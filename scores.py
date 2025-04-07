from utils import *  # Import all common utilities
import duckdb
from pydantic import BaseModel, Field, validator, ValidationError, field_validator
from fastapi import APIRouter, HTTPException, Body, Path, Query
from io import BytesIO
from typing import Optional, List
import pandas as pd
import requests
import numpy as np
import json
import logging
import ast
import time

# Disable the verbose Azure Storage logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)

router = APIRouter()

# Utility function to safely convert parameters to integers when possible
def safe_int_conversion(value):
    """
    Safely convert a value to integer if it's a digit string.
    Otherwise, return the original value.
    """
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value

# Helper function to validate batting order structure
def validate_batting_order(batting_order):
    if not isinstance(batting_order, dict):
        logger.warning("Batting order is not a dictionary")
        return False
    
    for order_num, inning_data in batting_order.items():
        if not isinstance(inning_data, dict):
            logger.warning(f"Inning data for order {order_num} is not a dictionary")
            return False
        
        for inning_num, player_data in inning_data.items():
            if not isinstance(player_data, dict):
                logger.warning(f"Player data for order {order_num}, inning {inning_num} is not a dictionary")
                return False
            
            # Check required keys
            for key in ['jersey_number', 'player_name', 'display']:
                if key not in player_data:
                    logger.warning(f"Missing required key '{key}' in player data for order {order_num}, inning {inning_num}")
                    return False
    
    return True

# Helper function to validate plate appearances structure
def validate_plate_appearances(plate_appearances):
    if not isinstance(plate_appearances, dict):
        logger.warning("Plate appearances is not a dictionary")
        return False
    
    for inning_num, inning_data in plate_appearances.items():
        if not isinstance(inning_data, dict) or 'rounds' not in inning_data:
            logger.warning(f"Invalid inning data for inning {inning_num}")
            return False
        
        rounds = inning_data.get('rounds', {})
        if not isinstance(rounds, dict):
            logger.warning(f"Rounds for inning {inning_num} is not a dictionary")
            return False
        
        for round_id, round_data in rounds.items():
            if not isinstance(round_data, dict):
                logger.warning(f"Round data for inning {inning_num}, round {round_id} is not a dictionary")
                return False
            
            if 'order_number' not in round_data or 'details' not in round_data:
                logger.warning(f"Missing required keys in round data for inning {inning_num}, round {round_id}")
                return False
            
            details = round_data.get('details', {})
            if not isinstance(details, dict):
                logger.warning(f"Details for inning {inning_num}, round {round_id} is not a dictionary")
                return False
            
            # Check minimum required keys for details
            # The full model would have many more fields, but we'll check a few critical ones
            for key in ['inning_number', 'order_number', 'pa_round', 'batter_seq_id']:
                if key not in details:
                    logger.warning(f"Missing required key '{key}' in details for inning {inning_num}, round {round_id}")
                    # Don't return False here as some fields might be optional
    
    return True

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
    bunt: int  # 1 for yes, 0 for no
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
    order_number: int
    batter_seq_id: int
    inning_number: int
    home_or_away: str = Field(...)
    batting_order_position: int
    team_id: int
    teamId: int
    game_id: int
    gameId: int
    out: int = Field(..., ge=0, le=1)
    my_team_ha: str = Field(...)
    wild_pitch: int = Field(..., ge=0, le=50)
    passed_ball: int = Field(..., ge=0, le=50)
    rbi: int = Field(..., ge=0, le=3)
    late_swings: int = Field(..., ge=0, le=5)
    qab: int = Field(..., ge=0, le=1)
    hard_hit: int = Field(..., ge=0, le=1)
    slap: int = Field(..., ge=0, le=1)
    bunt: int = Field(..., ge=0, le=1)
    sac: int = Field(..., ge=0, le=1)
    out_at: int = Field(..., ge=0, le=4)
    pa_why: Optional[str]
    pa_result: int = Field(..., ge=0, le=4)
    hit_to: int = Field(..., ge=0, le=9)
    br_result: int = Field(..., ge=0, le=4)
    pa_error_on: List[int] = []
    br_stolen_bases: List[int] = []
    base_running_hit_around: List[int] = []
    br_error_on: List[int] = []
    pitch_count: int = Field(..., ge=0, le=50)
    balls_before_play: int = Field(..., ge=0, le=3)
    strikes_before_play: int = Field(..., ge=0, le=2)
    strikes_unsure: int = Field(..., ge=0, le=2)
    strikes_watching: int = Field(..., ge=0, le=2)
    strikes_swinging: int = Field(..., ge=0, le=2)
    ball_swinging: int = Field(..., ge=0, le=2)
    fouls: int = Field(0, ge=0, le=50)  # Adding fouls field with default value

    @field_validator('home_or_away', 'my_team_ha')
    def validate_home_away(cls, v):
        if v not in {"home", "away"}:
            raise ValueError("must be 'home' or 'away'")
        return v

    @field_validator('pa_error_on', 'br_error_on')
    def validate_position_lists(cls, v, info):
        if not isinstance(v, list):
            if isinstance(v, (int, float)) and not pd.isna(v):
                v = int(v)
                if 0 <= v <= 9:
                    return [v]
            return []
        result = []
        for item in v:
            try:
                if item == "":
                    continue
                if isinstance(item, str) and item.isdigit():
                    val = int(item)
                    if 0 <= val <= 9:
                        result.append(val)
                elif isinstance(item, (int, float)) and not pd.isna(item):
                    val = int(item)
                    if 0 <= val <= 9:
                        result.append(val)
            except (ValueError, TypeError):
                pass
                
        return result
        
    @field_validator('br_stolen_bases', 'base_running_hit_around')
    def validate_base_lists(cls, v, info):
        if not isinstance(v, list):
            if isinstance(v, (int, float)) and not pd.isna(v):
                v = int(v)
                if 0 <= v <= 4:
                    return [v]
            return []
        result = []
        for item in v:
            try:
                if item == "":
                    continue
                if isinstance(item, str) and item.isdigit():
                    val = int(item)
                    if 0 <= val <= 4:
                        result.append(val)
                elif isinstance(item, (int, float)) and not pd.isna(item):
                    val = int(item)
                    if 0 <= val <= 4:
                        result.append(val)
            except (ValueError, TypeError):
                pass
                
        return result
###########################################################
#1 -  DELETE PLATE APPEARANCE
########################################################### 
@router.delete("/api/plate-appearance/{team_id}/{game_id}/{inning_number}/{team_choice}/{batter_seq_id}")
async def delete_plate_appearance(
    team_id: str,
    game_id: str, 
    inning_number: str,
    team_choice: str,
    batter_seq_id: str
):
    try:
        # Validate team_choice
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        
        # Convert team_id, game_id, inning_number, and batter_seq_id to integers if possible
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        inning_number_val = safe_int_conversion(inning_number)
        batter_seq_id_val = safe_int_conversion(batter_seq_id)
        
        # Create the blob path
        blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number_val}/{team_choice}_{batter_seq_id_val}.parquet"
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
# 2 - Plate appearance
###########################################################
@router.post("/api/plate-appearance", status_code=201)
#I wan to apply the pa_model to the data
async def save_plate_appearance(pa_data: dict = Body(...)):
    try:
        # Convert any string IDs to integers where appropriate
        if isinstance(pa_data, dict):
            for key in ['team_id', 'game_id', 'batter_seq_id', 'teamId', 'gameId']:
                if key in pa_data and isinstance(pa_data[key], str):
                    pa_data[key] = safe_int_conversion(pa_data[key])
        
        # Validate data using PlateAppearanceData model
        try:
            pa_model = PlateAppearanceData(**pa_data)
            pa_data = pa_model.model_dump()
        except ValidationError as e:
                print(e.errors())  # This will give you a structured list of errors.
                print(e.json())    # This will print the errors as JSON, which is great for debugging.
                raise HTTPException(
                    status_code=422,
                    detail=e.errors()  # you can even return structured error details instead of string
                )
        #print the validated data:
        print("------1 - VALIDATE PLATE APPEARANCE ------------")
        print(f"Validated data: {pa_data}")
        print("----------------------------------------------")
        # Convert to DataFrame
        df = pd.DataFrame([pa_data])
        team_data_key = "home" if pa_data["home_or_away"] == "home" else "away"
        blob_name = f"games/team_{pa_data['team_id']}/game_{pa_data['game_id']}/inning_{pa_data['inning_number']}/{team_data_key}_{pa_data['batter_seq_id']}.parquet"
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(parquet_buffer, overwrite=True)

        #print("------1A - SAVE PA DF (*) ------------")
        #print(f"calculate inning score batters_dfcolumns: {list(df.columns)}")
        #print(f"First row sample: {df.iloc[0].to_dict() if not df.empty else 'No data'}")
        #print("----------------------------------------------")
        
        return {
            "status": "success",
            "message": f"Plate appearance data saved for team {pa_data['team_id']}, game {pa_data['game_id']}, inning {pa_data['inning_number']}, batter sequence {pa_data['batter_seq_id']}",
            "blob_path": blob_name,
            "team_choice": pa_data["home_or_away"],
            "my_team_ha": pa_data.get("my_team_ha", "home")
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

###########################################################
# 3 - calculate the inning score
###########################################################
@router.post("/{team_id}/{game_id}/{inning_number}/{team_choice}/calculate-score")
async def calculate_score(team_id: str, game_id: str, inning_number: str, team_choice: str):
    try:
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        
        # Convert team_id, game_id, and inning_number to integers if possible using utility function  
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        inning_number_val = safe_int_conversion(inning_number)
        
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        plate_appearance_blob_name = f"games/team_{team_id_val}/game_{game_id_val}/inning_{inning_number_val}/{team_choice}_*.parquet"    
        # Continue with your existing logic for valid data
        batters_df = pd.DataFrame()
        try:
            # New improved query that directly aggregates all required columns
            aggregate_query = f"""
                WITH plate_appearances AS (
                    SELECT *,
                            CASE WHEN COALESCE(pitch_count, 0) > 0 THEN  
                            pitch_count - (COALESCE(CASE WHEN pa_why IN ('HBP','BB') THEN 1 ELSE 0 END, 0) +
                            COALESCE(balls_before_play, 0))
                            ELSE 0 END
                        AS total_strikes,
                        CASE WHEN pa_result > 0 THEN 1 ELSE 0 END AS on_base_count,
                        CASE 
                            WHEN (len(pa_error_on) + len(br_error_on)) > 0 THEN (len(pa_error_on) + len(br_error_on)) 
                            WHEN (len(pa_error_on) + len(br_error_on)) = 0 AND pa_why = 'E' THEN 1
                            ELSE 0 END AS errors,
                        1 as plate_appearances,
                        CASE WHEN br_result = 4 OR pa_result = 4 THEN 1 ELSE 0 END AS runs,
                        CASE WHEN pa_why in ['KK','K'] THEN 1 ELSE 0 END AS strikeouts,
                        CASE WHEN pa_why in ['BB','HBP'] THEN 1 ELSE 0 END AS walks,
                        CASE WHEN pa_why in ['H','HR','GS','S','B'] THEN 1 ELSE 0 END AS hits,
                        CASE WHEN (pa_result = 1 OR br_result = 1) AND NOT (br_result IN [2,3,4] OR out = 1 OR pa_result IN [0,2,3,4]) THEN 1 ELSE 0 END AS on_first_base,
                        CASE WHEN (pa_result = 2 OR br_result = 2) AND NOT (br_result IN [3,4] OR out = 1 OR pa_result IN [0,3,4]) THEN 1 ELSE 0 END AS on_second_base,
                        CASE WHEN (pa_result = 3 OR br_result = 3) AND NOT (br_result IN [4] OR out = 1 OR pa_result IN [0,4]) THEN 1 ELSE 0 END AS on_third_base,
                    FROM read_parquet('azure://{CONTAINER_NAME}/{plate_appearance_blob_name}', union_by_name=True)
                )
                            SELECT
                                team_id,
                                game_id,
                                inning_number,
                                home_or_away,
                                SUM(runs) AS runs,
                                SUM(hits) AS hits,
                                SUM(errors) AS errors,
                                SUM(strikeouts) AS strikeouts,
                                SUM(walks) AS walks,
                                SUM(out) AS out,
                                SUM(hard_hit) AS hard_hit,
                                SUM(on_first_base) AS on_first_base,
                                SUM(on_second_base) AS on_second_base,
                                SUM(on_third_base) AS on_third_base,
                                CAST(COALESCE(ROUND(SUM(total_strikes) * 100.0 / NULLIF(SUM(pitch_count), 0), 0), 0) AS INTEGER) AS strike_percent,
                                CAST(COALESCE(ROUND(SUM(on_base_count) * 100.0 / NULLIF(SUM(plate_appearances), 0), 0), 0) AS INTEGER) AS on_base_percent,
                                SUM(pitch_count) AS total_pitches,
                            FROM plate_appearances
                            GROUP BY team_id, game_id, inning_number, home_or_away
            """
            batters_df = con.execute(aggregate_query).fetchdf()
                
            #print("------2 - CALCULATE-INNING-SCORE (*) ------------")
            #print(f"calculate inning score batters_df columns: {list(batters_df.columns)}")
            #print(f"First row sample: {batters_df.iloc[0].to_dict() if not batters_df.empty else 'No data'}")
            #print("----------------------------------------------")
            
            if batters_df.empty:
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
# 4 - Read the box score from calc score
###########################################################
@router.get("/{team_id}/{game_id}/summary")
async def get_game_summary(team_id: str, game_id: str):
    """
    Get a summary of all innings for a game with box score format
    """
    try:
        con = get_duckdb_connection()
        
        # Convert team_id and game_id to integers if possible using utility function
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        
        # Initialize default response structure with blanks
        response = {
            "game_header": {
                "user_team": team_id_val, 
                "coach": "Unknown", 
                "opponent_name": "Unknown",
                "event_date": "", 
                "event_hour": 0,
                "event_minute": 0,
                "field_name": "",
                "field_location": "",
                "field_type": "",
                "field_temperature": 0,
                "game_status": "open",
                "my_team_ha": "home",
                "game_id": game_id_val
            },
            "innings": {str(i): {
                "home": {"runs": 0, "strike_percent": 0, "hard_hit": 0, "hits": 0, "outs": 0, "errors": 0, "walks": 0, "strikeouts": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0},
                "away": {"runs": 0, "strike_percent": 0, "hard_hit": 0, "hits": 0, "outs": 0, "errors": 0, "walks": 0, "strikeouts": 0, "on_first_base": 0, "on_second_base": 0, "on_third_base": 0}
            } for i in range(1, 8)},
            "totals": {
                "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "strikeouts": 0},
                "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "strikeouts": 0}
            }
        }
        
        # Step 1: Populate game header
        game_header_populated = False
        try:
            game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"       
            header_query = f"""
                SELECT 
                  {team_id} as user_team
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
                , COALESCE(game_status, 'open') as game_status
                , {game_id} as game_id
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}')
            """
            game_header_df = con.execute(header_query).fetchdf()
            if not game_header_df.empty:
                response["game_header"] = game_header_df.iloc[0].to_dict()
                game_header_populated = True
                logger.info(f"Game header populated for team {team_id}, game {game_id}")
                
                # Convert field_temperature to integer
                if "field_temperature" in response["game_header"]:
                    response["game_header"]["field_temperature"] = safe_int_conversion(response["game_header"]["field_temperature"])
        except Exception as e:
            logger.warning(f"Error getting game header: {str(e)}")
        
        # Step 2: Populate innings data
        try:
            game_innings_blob_name = f"games/team_{team_id}/game_{game_id}/box_score/*.parquet"              
            query = f"""
                SELECT 
                  home_or_away
                , inning_number
                , runs
                , strike_percent
                , hard_hit
                , hits
                , out as outs
                , walks
                , strikeouts
                , errors
                , on_first_base
                , on_second_base
                , on_third_base
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_innings_blob_name}', union_by_name=True)
            """
            innings_df = con.execute(query).fetchdf()
            
            if not innings_df.empty:
                # Track totals across innings
                totals = {
                    "home": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "strikeouts": 0},
                    "away": {"runs": 0, "hits": 0, "errors": 0, "walks": 0, "strikeouts": 0}
                }
                
                # Process each inning and update the response
                for _, row in innings_df.iterrows():
                    inning = int(row['inning_number'])
                    team = row['home_or_away'].lower()
                    if 1 <= inning <= 7:
                        inning_str = str(inning)
                        # Copy stats to the innings structure
                        for stat in ['runs', 'strike_percent', 'hard_hit', 'hits', 'outs', 'errors', 'on_first_base', 'on_second_base', 'on_third_base']:
                            if stat in row:
                                response["innings"][inning_str][team][stat] = int(row[stat])
                        # Add stats to totals
                        for stat in ['runs', 'hits', 'walks', 'strikeouts', 'errors']:
                            if stat in row:
                                totals[team][stat] += int(row[stat])
                        # Store strike_percent for swapping
                        if 'strike_percent' in row:
                            response["innings"][inning_str][team]['strike_percent'] = int(row['strike_percent'])
                
                # Update the totals in the response
                for team in ['home', 'away']:
                    for stat in ['runs', 'hits', 'walks', 'strikeouts', 'errors']:
                        response["totals"][team][stat] = totals[team][stat]
                
                # Step 3: Swap errors and strike_percent between teams
                for inning_str in response["innings"]:
                    # Swap errors - defensive team's errors are credited to the offensive team
                    home_errors = response["innings"][inning_str]["home"]["errors"]
                    away_errors = response["innings"][inning_str]["away"]["errors"]
                    response["innings"][inning_str]["home"]["errors"] = away_errors
                    response["innings"][inning_str]["away"]["errors"] = home_errors
                    
                    # Swap strike_percent - pitcher's strike percentage is credited to the defensive team
                    home_strike_percent = response["innings"][inning_str]["home"]["strike_percent"]
                    away_strike_percent = response["innings"][inning_str]["away"]["strike_percent"]
                    response["innings"][inning_str]["home"]["strike_percent"] = away_strike_percent
                    response["innings"][inning_str]["away"]["strike_percent"] = home_strike_percent
                    
                    # Swap strikeouts - pitcher's strikeouts are credited to the defensive team
                    home_strikeouts = response["innings"][inning_str]["home"]["strikeouts"]
                    away_strikeouts = response["innings"][inning_str]["away"]["strikeouts"]
                    response["innings"][inning_str]["home"]["strikeouts"] = away_strikeouts
                    response["innings"][inning_str]["away"]["strikeouts"] = home_strikeouts
                
                # Swap totals for errors
                home_total_errors = response["totals"]["home"]["errors"]
                away_total_errors = response["totals"]["away"]["errors"]
                response["totals"]["home"]["errors"] = away_total_errors
                response["totals"]["away"]["errors"] = home_total_errors
                
                # Swap totals for strikeouts
                home_total_strikeouts = response["totals"]["home"]["strikeouts"]
                away_total_strikeouts = response["totals"]["away"]["strikeouts"]
                response["totals"]["home"]["strikeouts"] = away_total_strikeouts
                response["totals"]["away"]["strikeouts"] = home_total_strikeouts
                
                logger.info(f"Innings data populated for team {team_id}, game {game_id}")
                
        except Exception as e:
            logger.warning(f"Error retrieving innings data: {str(e)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_game_summary: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # We'll still return the structure initialized at the beginning of the function
        return response


###########################################################
# 6 - Get plate appearance data for a specific inning only - FAST VERSION
###########################################################
@router.get("/fast/{team_id}/{game_id}/{team_choice}/{inning_number}/score_card_grid_one_inning")
async def get_inning_scorecardgrid_fast(team_id: str, game_id: str, team_choice: str, inning_number: str):
    """
    Get plate appearance data for a specific inning for a team.
    Uses a direct DuckDB query without nesting by pa_round for faster performance.
    """
    try:
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
        
        # Initialize DuckDB connection
        con = get_duckdb_connection()
        
        # Convert parameters to integers if possible
        team_id_val = safe_int_conversion(team_id)
        game_id_val = safe_int_conversion(game_id)
        inning_number_val = safe_int_conversion(inning_number)
        
        # Use a query that calculates pa_round directly in the source_data
        pa_duckdb_query = f"""
            WITH source_data AS (
                SELECT *,
                    -- Assign a dense rank to each unique order_number within this inning
                    -- This will be the pa_round for each order position
                    DENSE_RANK() OVER (
                        PARTITION BY order_number 
                        ORDER BY batter_seq_id
                    ) AS pa_round
                FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_{game_id_val}/inning_{inning_number_val}/{team_choice}_*.parquet')
            )
            SELECT
                json_object(
                        --PA header 
                        'team_id', {team_id_val},
                        'game_id', {game_id_val},
                        'team_choice', '{team_choice}',
                        'inning_number', {inning_number_val},
                        'pa_round', pa_round,
                        'batter_seq_id', batter_seq_id,
                        'order_number', order_number,

                        -- PA details
                        'pa_why', pa_why,
                        'pa_result', pa_result,
                        'hit_to', hit_to,
                        'out', out,
                        'out_at', out_at,
                        'balls_before_play', balls_before_play,
                        'strikes_before_play', strikes_before_play,
                        'pitch_count', pitch_count,
                        'strikes_unsure', strikes_unsure,
                        'strikes_watching', strikes_watching,
                        'strikes_swinging', strikes_swinging,
                        'ball_swinging', ball_swinging,
                        'fouls', fouls,
                        'hard_hit', hard_hit,
                        'late_swings', late_swings,
                        'slap', slap,
                        'bunt', bunt,
                        'qab', qab,
                        'rbi', rbi,
                        'br_result', br_result,
                        'wild_pitch', wild_pitch,
                        'passed_ball', passed_ball,
                        'sac', sac,
                        'br_stolen_bases', CASE WHEN br_stolen_bases IS NULL THEN '[]' ELSE json_array(br_stolen_bases) END,
                        'base_running_hit_around', CASE WHEN base_running_hit_around IS NULL THEN '[]' ELSE json_array(base_running_hit_around) END,
                        'pa_error_on', CASE WHEN pa_error_on IS NULL THEN '[]' ELSE json_array(pa_error_on) END,
                        'br_error_on', CASE WHEN br_error_on IS NULL THEN '[]' ELSE json_array(br_error_on) END
                ) AS details
            FROM source_data
            ORDER BY pa_round, order_number
        """
        
        # Execute the query and get results as JSON strings
        result_rows = con.execute(pa_duckdb_query).fetchall()
        
        # Handle empty results
        if not result_rows:
            return {
                "team_id": team_id_val,
                "game_id": game_id_val,
                "team_choice": team_choice,
                "inning_number": inning_number_val,
                "pa_available": "no",
                "scorebook_entries": []
            }
            
        # Parse each JSON string to a dictionary
        pa_details_list = []
        for row in result_rows:
            try:
                details_dict = json.loads(row[0])
                pa_details_list.append(details_dict)
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Error parsing JSON: {str(e)}")
                continue
        
        # Sort the entire list by order_number and then by pa_round
        sorted_pa_list = sorted(pa_details_list, key=lambda x: (int(x.get('order_number', 0)), int(x.get('pa_round', 0))))
        
        return {
            "team_id": team_id_val,
            "game_id": game_id_val,
            "team_choice": team_choice,
            "inning_number": inning_number_val,
            "pa_available": "yes" if sorted_pa_list else "no",
            "scorebook_entries": sorted_pa_list
        }
        
    except Exception as e:
        logger.error(f"Error in get_inning_scorecardgrid_fast: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logger.error(error_details)
        
        return {
            "team_id": team_id_val if 'team_id_val' in locals() else team_id,
            "game_id": game_id_val if 'game_id_val' in locals() else game_id,
            "team_choice": team_choice,
            "inning_number": inning_number_val if 'inning_number_val' in locals() else inning_number,
            "pa_available": "no",
            "scorebook_entries": [],
            "error": str(e)
        }

###########################################################
# 7 - Get plate appearance data for a specific inning only and batter_seq_id - NEW VERSION
###########################################################
@router.get("/new/{team_id}/{game_id}/{team_choice}/{inning_number}/scorecardgrid_paonly_inningonly/{batter_seq_id}/pa_edit")
async def get_inning_scorecardgrid_paonly_by_inning_new_pa_edit(team_id: str, game_id: str, team_choice: str, inning_number: str, batter_seq_id: str):
    """
    Get only plate appearance data for a specific inning for a team.
    This is a new version with a different URL path.
    """
    try:
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
    except Exception as e:
        logger.error(f"Error in get_inning_scorecardgrid_paonly_by_inning_new: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
        
    # Initialize DuckDB connection and get blob service client
    con = get_duckdb_connection()
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # Convert team_id, game_id, inning_number, and batter_seq_id to integers if possible using utility function
    team_id_val = safe_int_conversion(team_id)
    game_id_val = safe_int_conversion(game_id)
    inning_number_val = safe_int_conversion(inning_number)
    batter_seq_id_val = safe_int_conversion(batter_seq_id)
    
    ###########################################################
    # Get plate appearance data for specific inning
    ###########################################################
    try:
        pa_duckdb_query = f"""
            WITH src AS (
                SELECT 
                    *,
                    -- Calculate pa_round for each order_number within each inning
                    ROW_NUMBER() OVER (
                        PARTITION BY inning_number, order_number 
                        ORDER BY batter_seq_id
                    ) AS pa_round
                FROM read_parquet('azure://{CONTAINER_NAME}/games/team_{team_id_val}/game_{game_id_val}/inning_{inning_number_val}/{team_choice}_{batter_seq_id_val}.parquet')
                )
                SELECT
                    json_object(
                            --PA header 
                            'team_id', {team_id_val},
                            'game_id', {game_id_val},
                            'team_choice', '{team_choice}',
                            'inning_number', {inning_number_val},
                            'batter_seq_id', {batter_seq_id_val},
                            'order_number', order_number,
                            'pa_round', pa_round,
                            -- PA details
                            'pa_why', pa_why,
                            'pa_result', pa_result,
                            'hit_to', hit_to,
                            'out', out,
                            'out_at', out_at,
                            'balls_before_play', balls_before_play,
                            'strikes_before_play', strikes_before_play,
                            'pitch_count', pitch_count,
                            'strikes_unsure', strikes_unsure,
                            'strikes_watching', strikes_watching,
                            'strikes_swinging', strikes_swinging,
                            'ball_swinging', ball_swinging,
                            'fouls', fouls,
                            'hard_hit', hard_hit,
                            'late_swings', late_swings,
                            'slap', slap,
                            'bunt', bunt,
                            'qab', qab,
                            'rbi', rbi,
                            'br_result', br_result,
                            'wild_pitch', wild_pitch,
                            'passed_ball', passed_ball,
                            'sac', sac,
                            'br_stolen_bases', br_stolen_bases,
                            'base_running_hit_around', base_running_hit_around,
                            'pa_error_on', pa_error_on,
                            'br_error_on', br_error_on
                    ) AS details
                FROM src
        """
        start_time = time.time()
        pa_entries = con.execute(pa_duckdb_query).fetchone()
        query_time = time.time() - start_time
        logger.info(f"PA query execution time: {query_time:.3f} seconds")
        
        if pa_entries is None or len(pa_entries) == 0:
            pa_available = "no"
            pa_details = {}
        else:
            pa_available = "yes"
            # Parse the JSON string into a Python dict
            try:
                pa_details = json.loads(pa_entries[0])
            except (json.JSONDecodeError, IndexError, TypeError):
                pa_available = "no"
                pa_details = {}
    except Exception as e:
        logger.error(f"Error reading plate appearance info: {str(e)}")
        pa_available = "no"
        pa_details = {}
    
    ###########################################################
    # Process the details directly without nesting
    ###########################################################
    if pa_available == "yes" and isinstance(pa_details, dict):
        # Process array fields
        array_fields = ["pa_error_on", "br_error_on", "br_stolen_bases", "base_running_hit_around"]
        
        for field in array_fields:
            if field in pa_details:
                # Process the array field
                field_value = pa_details[field]
                
                # If not a list, convert to a list
                if not isinstance(field_value, list):
                    if field_value is None or field_value == "":
                        field_value = []
                    else:
                        # Try to convert string representations to list
                        if isinstance(field_value, str):
                            if field_value.startswith('[') and field_value.endswith(']'):
                                try:
                                    import ast
                                    field_value = ast.literal_eval(field_value)
                                except (SyntaxError, ValueError):
                                    # If parsing fails, treat as a single item
                                    field_value = [field_value]
                                else:
                                    field_value = [field_value]
                            else:
                                field_value = [field_value]
                        else:
                            field_value = [field_value]
                
                # Process each item in the list to ensure it's a properly typed integer
                if field in ["pa_error_on", "br_error_on"]:
                    # Baseball positions (0-9)
                    processed_items = []
                    for item in field_value:
                        try:
                            if isinstance(item, (int, float)) and not pd.isna(item):
                                val = int(item)
                                if 0 <= val <= 9:
                                    processed_items.append(val)
                            elif isinstance(item, str) and item.isdigit():
                                val = int(item)
                                if 0 <= val <= 9:
                                    processed_items.append(val)
                        except (ValueError, TypeError):
                            pass
                    pa_details[field] = processed_items
                elif field in ["br_stolen_bases", "base_running_hit_around"]:
                    # Base numbers (0-4)
                    processed_items = []
                    for item in field_value:
                        try:
                            if isinstance(item, (int, float)) and not pd.isna(item):
                                val = int(item)
                                if 0 <= val <= 4:
                                    processed_items.append(val)
                            elif isinstance(item, str) and item.isdigit():
                                val = int(item)
                                if 0 <= val <= 4:
                                    processed_items.append(val)
                        except (ValueError, TypeError):
                            pass
                    pa_details[field] = processed_items
        
        # Return the flat structure directly
        return pa_details
        
    # If no valid data, return a structured error response
    return {
        "team_id": team_id_val,
        "game_id": game_id_val,
        "team_choice": team_choice,
        "inning_number": inning_number_val,
        "batter_seq_id": batter_seq_id_val,
        "error": "No plate appearance data found"
    }
