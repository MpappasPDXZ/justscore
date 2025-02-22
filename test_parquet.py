from azure.storage.blob import BlobServiceClient
import pandas as pd
import duckdb
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_depth_chart_get():
    try:
        # Define position order
        position_order = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        
        # Get Azure credentials from environment variables
        ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
        ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        CONTAINER_NAME = "justscorecontainer"
        team_id = "1"
        
        # Connect to DuckDB
        con = duckdb.connect()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        con.execute("SET azure_transport_option_type = 'curl';")
        con.execute(f"SET azure_storage_connection_string='{connection_string}';")

        print("\nReading depth chart data...")
        # Read depth chart
        depth_chart_query = f"""
            SELECT 
                CAST(team_id AS INTEGER) as team_id,
                position,
                player_rank,
                jersey_number
            FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/depth_chart/depth_chart.parquet')
        """
        
        print("\nReading player data...")
        # Read player data
        player_query = f"""
            SELECT 
                jersey_number,
                player_name
            FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/*.parquet')
            WHERE jersey_number IS NOT NULL
        """
        
        # Execute queries
        depth_chart_df = con.execute(depth_chart_query).fetchdf()
        player_df = con.execute(player_query).fetchdf()
        
        print("\nDepth Chart Data:")
        print(depth_chart_df)
        
        print("\nPlayer Data:")
        print(player_df)
        
        # Merge dataframes
        result_df = pd.merge(depth_chart_df, player_df, on='jersey_number', how='left')
        
        # Create player_display column
        result_df['player_name'] = result_df.apply(
            lambda x: f"{x['jersey_number']} - {x['player_name']}", axis=1
        )
        
        # Sort by custom position order and then by rank
        result_df['position_order'] = result_df['position'].map({pos: idx for idx, pos in enumerate(position_order)})
        result_df = result_df.sort_values(['position_order', 'player_rank'])
        result_df = result_df.drop('position_order', axis=1)  # Remove the ordering column
        
        print("\nFinal Merged Result (in correct position order):")
        print(result_df)
        
        # Print some statistics
        print("\nStatistics:")
        print(f"Total positions: {len(result_df['position'].unique())}")
        print(f"Total players in depth chart: {len(result_df)}")
        print("\nPlayers by position (in correct order):")
        position_counts = result_df.groupby('position').size()
        # Reorder the position counts
        position_counts = position_counts.reindex(position_order)
        print(position_counts)
        
        return result_df
        
    except Exception as e:
        print(f"Error testing depth chart: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting depth chart test for team 1...")
    df = test_depth_chart_get()
    print("\nTest completed successfully!") 