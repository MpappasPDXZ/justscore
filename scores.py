from utils import *  # Import all common utilities
import duckdb
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from io import BytesIO
from typing import Optional, List
import pandas as pd
import requests

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
    bunt_or_slap: int  # 1 for bunt, 2 for slap, 0 for no
    base_running: str
    base_running_stolen_base: int

class InningScores(BaseModel):
    team_id: str
    game_id: str
    inning_number: int
    scores: List[ScoreData]

@router.get("/{team_id}/{game_id}/summary")
async def get_game_summary(team_id: str, game_id: str):
    """
    Get a summary of all innings for a game with box score format
    """
    try:
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        try:
            # Read the game info
            query = f"""
                SELECT *
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}')
            """
            game_info_df = con.execute(query).fetchdf()
            
            if game_info_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Game information not found for team {team_id}, game {game_id}"
                )
            
            # Extract relevant game information
            # Make sure it is lower case
            my_team_ha = game_info_df['my_team_ha'].values[0].lower() # 'home' or 'away'
            if my_team_ha == "home":
                home_team_name = "Your Team (Home)"
                away_team_name = game_info_df['away_team_name'].values[0] + " (Away)"
            else:
                home_team_name = game_info_df['away_team_name'].values[0] + " (Away)"
                away_team_name = "Your Team (Away)"
            
        except Exception as e:
            logger.error(f"Error reading game info: {str(e)}")
            # Default values if game info can't be read
            my_team_ha = "home"
            away_team_name = "Failure to Read (Away)"
            home_team_name = "Your Team (Home)"
        
        # List all inning files for the game
        prefix = f"games/team_{team_id}/game_{game_id}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        # Initialize data structure for innings
        innings_data = {}
        for inning in range(1, 8):
            innings_data[str(inning)] = {
                "away_team": {
                    "runs": 0,
                    "hits": 0,
                    "errors": 0,
                    "walks": 0,
                    "outs": 0
                },
                "home_team": {
                    "runs": 0,
                    "hits": 0,
                    "errors": 0,
                    "walks": 0,
                    "outs": 0
                }
            }
        
        # Initialize totals
        totals = {
            "away_team": {
                "runs": 0,
                "hits": 0,
                "errors": 0,
                "walks": 0,
                "outs": 0
            },
            "home_team": {
                "runs": 0,
                "hits": 0,
                "errors": 0,
                "walks": 0,
                "outs": 0
            }
        }
        
        # Process each inning file
        for blob in blobs:
            if "inning_" in blob.name and blob.name.endswith(".parquet"):
                try:
                    # Determine if this is home or away team data
                    if "/m_inning_" in blob.name:
                        # My team data
                        team_key = "home_team" if my_team_ha == "home" else "away_team"
                        inning_number = blob.name.split("m_inning_")[1].split(".")[0]
                    elif "/o_inning_" in blob.name:
                        # Opponent team data
                        team_key = "away_team" if my_team_ha == "home" else "home_team"
                        inning_number = blob.name.split("o_inning_")[1].split(".")[0]
                    else:
                        continue
                    
                    # Read the inning data
                    query = f"""
                        SELECT 
                            pa_result,
                            detailed_result,
                            base_running
                        FROM read_parquet('azure://{CONTAINER_NAME}/{blob.name}')
                    """
                    inning_df = con.execute(query).fetchdf()
                    
                    if not inning_df.empty:
                        # Calculate runs, hits, errors, walks, and outs
                        runs = len(inning_df[inning_df['pa_result'] == 'HR']) + len(inning_df[inning_df['base_running'] == '4B'])
                        hits = len(inning_df[inning_df['pa_result'].isin(['1B', '2B', '3B', 'HR'])])
                        errors = len(inning_df[inning_df['pa_result'] == 'E'])
                        walks = len(inning_df[inning_df['pa_result'].isin(['BB', 'HB'])])
                        outs = len(inning_df[inning_df['pa_result'].isin(['K', 'KK', 'FO'])])
                        
                        # Update the inning stats (only if inning number is valid)
                        if inning_number in innings_data:
                            innings_data[inning_number][team_key] = {
                                "runs": runs,
                                "hits": hits,
                                "errors": errors,
                                "walks": walks,
                                "outs": outs
                            }
                        
                        # Update totals
                        totals[team_key]["runs"] += runs
                        totals[team_key]["hits"] += hits
                        totals[team_key]["errors"] += errors
                        totals[team_key]["walks"] += walks
                        totals[team_key]["outs"] += outs
                except Exception as e:
                    logger.warning(f"Error processing {blob.name}: {str(e)}")
                    continue
        
        # Build the final summary structure
        summary = {
            "away_team_name": away_team_name,
            "home_team_name": home_team_name,
            "innings": {}
        }
        
        # Add innings data
        for inning_number, inning_data in innings_data.items():
            summary["innings"][inning_number] = {
                "away_team": inning_data["away_team"],
                "home_team": inning_data["home_team"]
            }
        
        # Add totals
        summary["totals"] = {
            "away_team": totals["away_team"],
            "home_team": totals["home_team"]
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error getting game summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting game summary: {str(e)}"
        ) 

@router.get("/{team_id}/{game_id}/inning/{inning_number}/scorebook/{team_choice}")
async def get_inning_scorebook(team_id: str, game_id: str, inning_number: int, team_choice: str):
    """
    Get detailed scorebook-style data for a specific inning, joining batting results with lineup information
    team_choice: 'home' or 'away' - which team's data to retrieve
    """
    try:
        # Validate team_choice
        if team_choice not in ['home', 'away']:
            raise HTTPException(
                status_code=400,
                detail="team_choice must be 'home' or 'away'"
            )
            
        # Step 1: Get game info to determine home/away status
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        #game info
        try:
            query = f"""
                SELECT 
                    my_team_ha,
                    game_id,
                    user_team,
                    coach,
                    away_team_name,
                    event_date,
                    event_hour,
                    event_minute,
                    field_name,
                    field_location,
                    field_type,
                    field_temperature,
                    game_status
                FROM read_parquet('azure://{CONTAINER_NAME}/{game_info_blob_name}')
            """
            game_info_df = con.execute(query).fetchdf()
            if game_info_df.empty:
                print("no game info found")
                raise HTTPException(
                    status_code=404,
                    detail=f"Game information not found for team {team_id}, game {game_id}"
                )
            
           
        except Exception as e:
            logger.error(f"Error reading game info: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading game info: {str(e)}"
            )
        # Step 2 determine if my team is home or away.  team_choice is passed from the front end.  my_team_ha is from the game info
        my_team_ha = game_info_df['my_team_ha'].values[0].lower()  # 'home' or 'away'
        if (my_team_ha == "home" and team_choice == "home") or (my_team_ha == "away" and team_choice == "away"):
            team_data_key = 'm'
        else:
            team_data_key = 'o'
        #lineup info
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
                # Convert lineup dataframe to list of dictionaries for the response
                lineup_entries = lineup_df.to_dict(orient='records')
                
        except Exception as e:
            logger.error(f"Error reading lineup info: {str(e)}")
            lineup_available = False
            lineup_entries = []
            
        # Step 3: Get at bat data if there is any.
        batter_info_blob_name = f"games/team_{team_id}/game_{game_id}/{team_data_key}_inning_{inning_number}.parquet"
        try:
            query = f"""
                SELECT                     
                    batter_seq_id,
                    batter_jersey_number,
                    pa_result,
                    detailed_result,
                    base_running,
                    balls_before_play,
                    strikes_before_play,
                    fouls_after_two_strikes,
                    hard_hit,
                    bunt_or_slap,
                    base_running_stolen_base
                FROM read_parquet('azure://{CONTAINER_NAME}/{batter_info_blob_name}')
            """
            batter_df = con.execute(query).fetchdf()
            if batter_df.empty:
                print("no at bat data found")
                raise HTTPException(
                    status_code=404,
                    detail=f"no at bat data found"
                )           
        except Exception as e:
            logger.error(f"Error reading lineup info: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading lineup info: {str(e)}"
            )
        # Create a dictionary mapping from jersey numbers to names
        jersey_number_to_name = dict(zip(lineup_df['jersey_number'], lineup_df['name']))
        jersey_number_to_position = dict(zip(lineup_df['jersey_number'], lineup_df['position']))
        jersey_number_to_order = dict(zip(lineup_df['jersey_number'], lineup_df['order_number']))

        scorebook_entries = []
        for _, row in batter_df.iterrows():
            jersey_number = str(row["batter_jersey_number"])
            
            # Try to find the player name by jersey number
            if jersey_number in jersey_number_to_name:
                batter_name = jersey_number_to_name[jersey_number]
                position = jersey_number_to_position[jersey_number]
                order_number = jersey_number_to_order[jersey_number]
            else:
                # If jersey number not found, use a placeholder
                batter_name = f"Player #{jersey_number}"
                position = "Unknown"
                order_number = 0
            
            entry = {
                "order_number": order_number,
                "batter_jersey_number": jersey_number,
                "batter_name": batter_name,
                "batter_seq_id": int(row["batter_seq_id"]),               
                "pa_result": str(row["pa_result"]),
                "detailed_result": str(row["detailed_result"]),
                "base_running": str(row["base_running"]),
                "balls_before_play": int(row["balls_before_play"]),
                "strikes_before_play": int(row["strikes_before_play"]),
                "fouls_after_two_strikes": int(row["fouls_after_two_strikes"]),
                "hard_hit": int(row["hard_hit"]),
                "bunt_or_slap": int(row["bunt_or_slap"]),
                "base_running_stolen_base": int(row["base_running_stolen_base"])
            }
            
            scorebook_entries.append(entry)
            
        # Step 6: Build the response
        response = {
            "team_id": team_id,
            "game_id": game_id,
            "inning_number": inning_number,
            "team_choice": team_choice,
            "my_team_ha": my_team_ha,
            "lineup_available": lineup_available,
            "lineup_entries": lineup_entries,  # Add the lineup entries to the response
            "scorebook_entries": scorebook_entries

        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting inning scorebook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting inning scorebook: {str(e)}"
        )
    

