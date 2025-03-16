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
ARRAY_FIELDS = ["hit_around_bases", "stolen_bases"]

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
    why_base_reached: Optional[str] = ""
    hard_hit: Optional[int] = 0
    bunt_or_slap: Optional[int] = 0
    base_running_stolen_base: Optional[int] = 0
    stolen_bases: List[int] = []
    hit_around_bases: List[int] = []

@router.get("/{team_id}/{game_id}/box-score")
async def get_box_score(team_id: str, game_id: str):
    """
    Get a detailed box score summary for a game, including player statistics for both teams
    """
    try:
        con = get_duckdb_connection()
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Step 1: Get game info
        game_info_blob_name = f"games/team_{team_id}/game_{game_id}.parquet"
        try:
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
            my_team_ha = str(game_info_df['my_team_ha'].iloc[0]).lower()  # 'home' or 'away'
            away_team_name = game_info_df['away_team_name'].iloc[0]
            my_team_name = game_info_df['user_team'].iloc[0]
            
            # Set team names based on home/away status
            if my_team_ha == "home":
                home_team_name = my_team_name
            else:
                home_team_name = away_team_name
                away_team_name = my_team_name
                
        except Exception as e:
            logger.error(f"Error reading game info: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading game info: {str(e)}"
            )
        
        # Step 2: Get lineup information for both teams
        home_lineup = []
        away_lineup = []
        
        # My team lineup
        my_lineup_blob_name = f"games/team_{team_id}/game_{game_id}/m_lineup.parquet"
        try:
            query = f"""
                SELECT 
                    jersey_number,
                    name,
                    position,
                    order_number
                FROM read_parquet('azure://{CONTAINER_NAME}/{my_lineup_blob_name}')
                ORDER BY order_number
            """
            my_lineup_df = con.execute(query).fetchdf()
            
            if not my_lineup_df.empty:
                my_lineup = my_lineup_df.to_dict(orient='records')
                if my_team_ha == "home":
                    home_lineup = my_lineup
                else:
                    away_lineup = my_lineup
                    
        except Exception as e:
            logger.warning(f"Error reading my team lineup: {str(e)}")
        
        # Opponent lineup
        opp_lineup_blob_name = f"games/team_{team_id}/game_{game_id}/o_lineup.parquet"
        try:
            query = f"""
                SELECT 
                    jersey_number,
                    name,
                    position,
                    order_number
                FROM read_parquet('azure://{CONTAINER_NAME}/{opp_lineup_blob_name}')
                ORDER BY order_number
            """
            opp_lineup_df = con.execute(query).fetchdf()
            
            if not opp_lineup_df.empty:
                opp_lineup = opp_lineup_df.to_dict(orient='records')
                if my_team_ha == "home":
                    away_lineup = opp_lineup
                else:
                    home_lineup = opp_lineup
                    
        except Exception as e:
            logger.warning(f"Error reading opponent lineup: {str(e)}")
        
        # Step 3: Get all plate appearance data for all innings
        prefix = f"games/team_{team_id}/game_{game_id}/inning_"
        
        # List all blobs with the prefix
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        # Initialize player stats dictionaries
        home_player_stats = {}
        away_player_stats = {}
        
        # Initialize inning-by-inning scoring
        innings_scoring = {}
        for inning in range(1, 8):  # Assuming max 7 innings
            innings_scoring[str(inning)] = {
                "home": 0,
                "away": 0
            }
        
        # Process each plate appearance file
        for blob in blobs:
            try:
                # Parse the blob name to get inning and team info
                blob_parts = blob.name.split('/')
                filename = blob_parts[-1]
                
                # Skip if not a plate appearance file
                if not (filename.startswith('home_') or filename.startswith('away_')):
                    continue
                
                # Determine team and inning
                team_choice = "home" if filename.startswith('home_') else "away"
                inning_number = blob_parts[-2].split('_')[-1]
                
                # Read the plate appearance data
                query = f"""
                    SELECT *
                    FROM read_parquet('azure://{CONTAINER_NAME}/{blob.name}')
                """
                pa_df = con.execute(query).fetchdf()
                
                if pa_df.empty:
                    continue
                
                # Get player info
                order_number = int(pa_df['order_number'].iloc[0]) if 'order_number' in pa_df.columns else 0
                
                # Get or create player stats
                player_stats_dict = home_player_stats if team_choice == "home" else away_player_stats
                
                if order_number not in player_stats_dict:
                    # Find player name from lineup
                    player_name = f"Player #{order_number}"
                    jersey_number = "0"
                    position = ""
                    
                    lineup = home_lineup if team_choice == "home" else away_lineup
                    for player in lineup:
                        if player['order_number'] == order_number:
                            player_name = player['name']
                            jersey_number = player['jersey_number']
                            position = player['position']
                            break
                    
                    # Initialize player stats
                    player_stats_dict[order_number] = {
                        "name": player_name,
                        "jersey_number": jersey_number,
                        "position": position,
                        "order_number": order_number,
                        "AB": 0,  # At bats
                        "R": 0,   # Runs
                        "H": 0,   # Hits
                        "2B": 0,  # Doubles
                        "3B": 0,  # Triples
                        "HR": 0,  # Home runs
                        "RBI": 0, # Runs batted in
                        "BB": 0,  # Walks
                        "SO": 0,  # Strikeouts
                        "SB": 0,  # Stolen bases
                        "CS": 0,  # Caught stealing
                        "AVG": .000,  # Batting average
                        "OBP": .000,  # On-base percentage
                        "SLG": .000,  # Slugging percentage
                        "OPS": .000,  # On-base plus slugging
                        "PA": 0,  # Plate appearances
                        "HBP": 0, # Hit by pitch
                        "SF": 0,  # Sacrifice flies
                        "E": 0    # Errors
                    }
                
                # Update player stats based on plate appearance
                player_stats = player_stats_dict[order_number]
                player_stats["PA"] += 1
                
                # Process plate appearance result
                pa_result = pa_df['pa_result'].iloc[0] if 'pa_result' in pa_df.columns else ""
                
                # Update inning scoring if run scored
                if pa_result == "HR" or (pa_df['base_running'].iloc[0] == "4B" if 'base_running' in pa_df.columns else False):
                    innings_scoring[inning_number][team_choice] += 1
                    player_stats["R"] += 1
                
                # Update stats based on result
                if pa_result in ["1B", "2B", "3B", "HR"]:
                    player_stats["AB"] += 1
                    player_stats["H"] += 1
                    
                    if pa_result == "2B":
                        player_stats["2B"] += 1
                    elif pa_result == "3B":
                        player_stats["3B"] += 1
                    elif pa_result == "HR":
                        player_stats["HR"] += 1
                        player_stats["RBI"] += 1  # Assume at least 1 RBI for HR
                
                elif pa_result in ["K", "KK", "FO"]:
                    player_stats["AB"] += 1
                    if pa_result in ["K", "KK"]:
                        player_stats["SO"] += 1
                
                elif pa_result == "BB":
                    player_stats["BB"] += 1
                
                elif pa_result == "HB":
                    player_stats["HBP"] += 1
                
                elif pa_result == "E":
                    player_stats["AB"] += 1
                    player_stats["E"] += 1
                
                # Check for stolen bases
                if 'stolen_bases' in pa_df.columns and not pd.isna(pa_df['stolen_bases'].iloc[0]):
                    stolen_bases = pa_df['stolen_bases'].iloc[0]
                    if isinstance(stolen_bases, (list, np.ndarray)):
                        if hasattr(stolen_bases, 'tolist'):
                            stolen_bases = stolen_bases.tolist()
                        else:
                            stolen_bases = list(stolen_bases)
                        player_stats["SB"] += len(stolen_bases)
                
                # Calculate batting average, OBP, SLG, OPS
                if player_stats["AB"] > 0:
                    player_stats["AVG"] = round(player_stats["H"] / player_stats["AB"], 3)
                    
                    # Calculate OBP: (H + BB + HBP) / (AB + BB + HBP + SF)
                    obp_numerator = player_stats["H"] + player_stats["BB"] + player_stats["HBP"]
                    obp_denominator = player_stats["AB"] + player_stats["BB"] + player_stats["HBP"] + player_stats["SF"]
                    player_stats["OBP"] = round(obp_numerator / obp_denominator, 3) if obp_denominator > 0 else .000
                    
                    # Calculate SLG: (1B + 2*2B + 3*3B + 4*HR) / AB
                    singles = player_stats["H"] - player_stats["2B"] - player_stats["3B"] - player_stats["HR"]
                    total_bases = singles + (2 * player_stats["2B"]) + (3 * player_stats["3B"]) + (4 * player_stats["HR"])
                    player_stats["SLG"] = round(total_bases / player_stats["AB"], 3)
                    
                    # Calculate OPS: OBP + SLG
                    player_stats["OPS"] = round(player_stats["OBP"] + player_stats["SLG"], 3)
                
            except Exception as e:
                logger.warning(f"Error processing {blob.name}: {str(e)}")
                continue
        
        # Convert player stats dictionaries to lists sorted by order number
        home_players = [stats for _, stats in sorted(home_player_stats.items(), key=lambda x: x[0])]
        away_players = [stats for _, stats in sorted(away_player_stats.items(), key=lambda x: x[0])]
        
        # Calculate team totals
        home_totals = calculate_team_totals(home_players)
        away_totals = calculate_team_totals(away_players)
        
        # Build the final box score
        box_score = {
            "game_info": {
                "team_id": team_id,
                "game_id": game_id,
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "date": game_info_df['event_date'].iloc[0] if 'event_date' in game_info_df.columns else "",
                "location": game_info_df['field_name'].iloc[0] if 'field_name' in game_info_df.columns else ""
            },
            "innings_scoring": innings_scoring,
            "home_team": {
                "players": home_players,
                "totals": home_totals
            },
            "away_team": {
                "players": away_players,
                "totals": away_totals
            }
        }
        
        return box_score
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating box score: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Detailed error: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating box score: {str(e)}"
        )

def calculate_team_totals(players):
    """Calculate team totals from player statistics"""
    totals = {
        "AB": 0,
        "R": 0,
        "H": 0,
        "2B": 0,
        "3B": 0,
        "HR": 0,
        "RBI": 0,
        "BB": 0,
        "SO": 0,
        "SB": 0,
        "CS": 0,
        "AVG": .000,
        "OBP": .000,
        "SLG": .000,
        "OPS": .000,
        "PA": 0,
        "HBP": 0,
        "SF": 0,
        "E": 0
    }
    
    # Sum up all numeric fields
    for player in players:
        for key in totals.keys():
            if key in ["AVG", "OBP", "SLG", "OPS"]:
                continue  # Skip average calculations for now
            totals[key] += player[key]
    
    # Calculate team batting average, OBP, SLG, OPS
    if totals["AB"] > 0:
        totals["AVG"] = round(totals["H"] / totals["AB"], 3)
        
        # Calculate OBP: (H + BB + HBP) / (AB + BB + HBP + SF)
        obp_numerator = totals["H"] + totals["BB"] + totals["HBP"]
        obp_denominator = totals["AB"] + totals["BB"] + totals["HBP"] + totals["SF"]
        totals["OBP"] = round(obp_numerator / obp_denominator, 3) if obp_denominator > 0 else .000
        
        # Calculate SLG: (1B + 2*2B + 3*3B + 4*HR) / AB
        singles = totals["H"] - totals["2B"] - totals["3B"] - totals["HR"]
        total_bases = singles + (2 * totals["2B"]) + (3 * totals["3B"]) + (4 * totals["HR"])
        totals["SLG"] = round(total_bases / totals["AB"], 3)
        
        # Calculate OPS: OBP + SLG
        totals["OPS"] = round(totals["OBP"] + totals["SLG"], 3)
    
    return totals

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
                            [paste the column list here]
                        FROM read_parquet('azure://{CONTAINER_NAME}/{blob.name}', union_by_name=True)
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

@router.get("/{team_id}/{game_id}/{inning_number}/{team_choice}")
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
        
        # Get game info
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
            
        # Determine if my team is home or away - get the first value as a string
        my_team_ha = str(game_info_df['my_team_ha'].iloc[0]).lower()  # 'home' or 'away'
        if (my_team_ha == "home" and team_choice == "home") or (my_team_ha == "away" and team_choice == "away"):
            team_data_key = 'm'
        else:
            team_data_key = 'o'
            
        # Get lineup info
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
            
        # Step 3: Get at bat data from the new file structure
        # First, list all plate appearance files for this inning
        prefix = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/"
        
        # Determine which team's data to read based on team_choice
        file_prefix = "home_" if team_choice == "home" else "away_"
        
        # List all blobs with the prefix
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        # Filter blobs to only include those for the selected team
        team_blobs = [blob for blob in blobs if blob.name.split('/')[-1].startswith(file_prefix)]
        
        # If no blobs found, return empty scorebook entries instead of an error
        if not team_blobs:
            print(f"No at bat data found for team {team_id}, game {game_id}, inning {inning_number}, team {team_choice}")
            return {
                "team_id": team_id,
                "game_id": game_id,
                "inning_number": inning_number,
                "team_choice": team_choice,
                "my_team_ha": my_team_ha,
                "lineup_available": lineup_available,
                "lineup_entries": lineup_entries,
                "scorebook_entries": []  # Return empty list instead of error
            }

        batter_dfs = []
        for blob in team_blobs:
            try:
                # First, get the schema of the file to see what columns are available
                schema_query = f"""
                    SELECT * FROM read_parquet('azure://{CONTAINER_NAME}/{blob.name}') LIMIT 0
                """
                schema_df = con.execute(schema_query).fetchdf()
                available_columns = schema_df.columns.tolist()
                
                # Execute the query with only the available columns
                query = f"""
                    SELECT                        
                        order_number,
                        batter_seq_id,
                        inning_number,
                        home_or_away,
                        batting_order_position,
                        team_id,
                        game_id,
                        out_at,
                        balls_before_play,
                        strikes_before_play,
                        pitch_count,
                        strikes_unsure,
                        out,
                        br_result,
                        why_base_reached,
                        wild_pitch,
                        passed_ball,
                        strikes_watching,
                        strikes_swinging,
                        ball_swinging,
                        fouls,
                        fouls_after_two_strikes,
                        pa_result,
                        hit_to,
                        pa_why,
                        pa_error_on,
                        br_stolen_bases,
                        br_error_on,
                        base_running_hit_around,
                        base_running_other,
                        stolen_bases,
                        hit_around_bases,
                        teamId,
                        gameId,
                        my_team_ha
                    FROM read_parquet('azure://{CONTAINER_NAME}/{blob.name}', union_by_name=True)
                """
                df = con.execute(query).fetchdf()
                
                # Make sure required columns exist (add them if they don't)
                required_columns = ["batter_seq_id", "order_number"]
                for col in required_columns:
                    if col not in df.columns:
                        # Extract sequence ID from filename if possible
                        if col == "batter_seq_id":
                            try:
                                seq_id = int(blob.name.split('_')[-1].split('.')[0])
                                df[col] = seq_id
                            except:
                                df[col] = 0
                        else:
                            df[col] = 0
                
                batter_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {blob.name}: {str(e)}")
                continue
        
        if not batter_dfs:
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
        
        # Combine all dataframes
        batter_df = pd.concat(batter_dfs, ignore_index=True)
        
        # Sort by batter_seq_id to maintain order
        batter_df = batter_df.sort_values('batter_seq_id')
        
        # Create dictionaries for lookup
        order_to_jersey = {}
        order_to_name = {}
        order_to_position = {}
        
        # Only create these mappings if lineup data is available
        if lineup_available and not lineup_df.empty:
            order_to_jersey = dict(zip(lineup_df['order_number'], lineup_df['jersey_number']))
            order_to_name = dict(zip(lineup_df['order_number'], lineup_df['name']))
            order_to_position = dict(zip(lineup_df['order_number'], lineup_df['position']))

        # Create a dictionary to track rounds for each player
        player_rounds = {}
        
        # First pass: determine rounds for each player
        for _, row in batter_df.iterrows():
            # Get the order number from the plate appearance data
            order_number = int(row.get("order_number", 0))
            if hasattr(order_number, '__len__') and len(order_number) > 0:
                order_number = int(order_number[0])
            
            # Use order_number as the player identifier
            player_id = order_number
            
            # Initialize the round counter for this player if not exists
            if player_id not in player_rounds:
                player_rounds[player_id] = 0
            
            # Increment the round for this player
            player_rounds[player_id] += 1
        
        scorebook_entries = []
        # Second pass: create entries with round information
        for _, row in batter_df.iterrows():
            # Get the order number from the plate appearance data
            order_number = int(row.get("order_number", 0))
            if hasattr(order_number, '__len__') and len(order_number) > 0:
                order_number = int(order_number[0])
            
            # Use order_number as the player identifier
            player_id = order_number
            
            # Get the round for this plate appearance
            # We need to determine which round this is for the player
            # based on the batter_seq_id
            
            # Get all batter_seq_ids for this player
            player_seq_ids = batter_df[batter_df['order_number'] == player_id]['batter_seq_id'].tolist()
            
            # Sort them to determine the order
            player_seq_ids.sort()
            
            # Find the position of the current batter_seq_id in the sorted list
            current_seq_id = int(row['batter_seq_id'])
            round_number = player_seq_ids.index(current_seq_id) + 1  # 1-based indexing for rounds
            
            # Try to find the player info by order number
            if lineup_available and order_number in order_to_name:
                batter_name = order_to_name[order_number]
                jersey_number = order_to_jersey[order_number]
                position = order_to_position[order_number]
            else:
                # If order number not found in lineup, use data from plate appearance
                batter_name = row.get("batter_name", f"Player #{order_number}")
                if hasattr(batter_name, '__len__') and len(batter_name) > 0:
                    batter_name = str(batter_name[0])
                
                jersey_number = row.get("batter_jersey_number", "0")
                if hasattr(jersey_number, '__len__') and len(jersey_number) > 0:
                    jersey_number = str(jersey_number[0])
                
                position = "Unknown"
            
            # Create entry with all available fields
            entry = {
                "order_number": order_number,
                "batter_jersey_number": jersey_number,
                "batter_name": batter_name,
                "position": position,
                "batter_seq_id": int(row["batter_seq_id"]),
                "round": round_number  # Add the round number
            }
            
            # Add all other fields that exist in the data
            for field in batter_df.columns:
                if field not in entry and field in row:
                    value = row[field]
                    
                    # Special handling for known array fields
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
                                logger.warning(f"Failed to parse {field} as JSON: {str(e)}")
                                # If parsing fails, use empty list
                                entry[field] = []
                        else:
                            # Default to empty list for array fields
                            entry[field] = []
                            logger.warning(f"Using empty list for {field}, original type: {type(value)}")
                    else:
                        # Handle scalar values as before
                        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)) and len(value) > 0:
                            value = value[0]
                        
                        # Convert to appropriate type
                        if pd.api.types.is_numeric_dtype(type(value)):
                            entry[field] = int(value) if pd.notna(value) else 0
                        else:
                            entry[field] = str(value) if pd.notna(value) else ""
            
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
        required_fields = ["teamId", "gameId", "inning_number", "home_or_away", "batter_seq_id"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required field: {field}"
                )
        
        # Handle br_error_on if it's a list
        if isinstance(data.get('br_error_on'), list) and len(data['br_error_on']) > 0:
            data['br_error_on'] = str(data['br_error_on'][0])
        elif isinstance(data.get('br_error_on'), list) and len(data['br_error_on']) == 0:
            data['br_error_on'] = "0"
            
        # Ensure all required fields have values (even if empty)
        # This prevents errors when creating the DataFrame
        default_fields = {
            "pa_why": "",
            "hit_to": "0",
            "pa_error_on": "0",
            "br_error_on": "0",
            "passed_ball": 0,
            "ball_swinging": 0,
            "base_running_hit_around": 0,
            "base_running_other": 0
        }
        
        for field, default_value in default_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        
        # Normalize pa_why field to use the standard sequence: H, HH, B, BB, S, B, FC, E
        if 'pa_why' in data and data['pa_why']:
            pa_why_mapping = {
                'H': 'H',    # Hit
                'HH': 'HH',  # Hard Hit
                'B': 'B',    # Ball
                'BB': 'BB',  # Base on Balls
                'S': 'S',    # Strike
                'FC': 'FC',  # Fielder's Choice
                'E': 'E',    # Error
                # Add any other mappings needed
            }
            
            # If the value exists in the mapping, use the standardized version
            if data['pa_why'] in pa_why_mapping:
                data['pa_why'] = pa_why_mapping[data['pa_why']]
        
        # Ensure array fields are properly formatted
        for field in ARRAY_FIELDS:
            if field in data:
                # Make sure it's a list
                if not isinstance(data[field], list):
                    try:
                        # Try to convert to list if it's not already
                        data[field] = list(data[field])
                    except:
                        # If conversion fails, use empty list
                        data[field] = []
                
                # Log the array field for debugging
                logger.info(f"Saving {field} as: {data[field]} (type: {type(data[field])})")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Determine the team_data_key based on home_or_away
        team_data_key = "home" if data["home_or_away"] == "home" else "away"
        
        # Create the blob path
        blob_name = f"games/team_{data['teamId']}/game_{data['gameId']}/inning_{data['inning_number']}/{team_data_key}_{data['batter_seq_id']}.parquet"
        
        # Convert DataFrame to Parquet and save to memory
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_buffer.seek(0)
        
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Upload to Azure Blob Storage
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(parquet_buffer, overwrite=True)
        
        return {
            "status": "success",
            "message": f"Plate appearance data saved for team {data['teamId']}, game {data['gameId']}, inning {data['inning_number']}, batter sequence {data['batter_seq_id']}",
            "blob_path": blob_name
        }
    except HTTPException:
        raise
    except Exception as e:
        # Print the full error for debugging
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
    - team_id: The team ID
    - game_id: The game ID
    - inning_number: The inning number
    - team_choice: 'home' or 'away'
    - batter_seq_id: The batter sequence ID
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

# Helper function to check if a blob exists
def blob_exists(blob_client):
    try:
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False

@router.get("/debug/{team_id}/{game_id}/{inning_number}/{team_choice}/{batter_seq_id}")
async def debug_plate_appearance(team_id: str, game_id: str, inning_number: int, team_choice: str, batter_seq_id: int):
    """Debug endpoint to examine a specific plate appearance file"""
    try:
        # Determine the team_data_key based on team_choice
        team_data_key = "home" if team_choice == "home" else "away"
        
        # Create the blob path
        blob_name = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/{team_data_key}_{batter_seq_id}.parquet"
        
        # Get blob service client
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Check if the blob exists
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_exists(blob_client):
            return {
                "status": "error",
                "message": f"File not found: {blob_name}"
            }
        
        # Download the blob to memory
        blob_data = blob_client.download_blob().readall()
        
        # Use pandas to read the Parquet data
        buffer = BytesIO(blob_data)
        df = pd.read_parquet(buffer)
        
        # Convert to dictionary for inspection
        data_dict = df.to_dict(orient='records')[0] if not df.empty else {}
        
        # Special handling for array fields
        for field in ARRAY_FIELDS:
            if field in data_dict:
                value = data_dict[field]
                logger.info(f"DEBUG: Field {field}, Type: {type(value)}, Value: {value}")
                
                # Try different methods to convert to list
                if isinstance(value, (pd.Series, np.ndarray)):
                    data_dict[field] = value.tolist() if hasattr(value, 'tolist') else list(value)
                elif isinstance(value, str):
                    try:
                        data_dict[field] = json.loads(value)
                    except:
                        pass
        
        return {
            "status": "success",
            "file_path": blob_name,
            "data": data_dict,
            "column_types": {col: str(type(df[col].iloc[0])) for col in df.columns if not df.empty}
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "message": f"Error reading Parquet file: {str(e)}",
            "details": error_details
        }


