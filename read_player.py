import duckdb
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def read_player_parquet(team_id: str, jersey_numbers: list):
    try:
        # Get Azure credentials from environment variables
        ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
        ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        CONTAINER_NAME = "justscorecontainer"
        
        # Connect to DuckDB
        con = duckdb.connect()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        con.execute("SET azure_transport_option_type = 'curl';")
        con.execute(f"SET azure_storage_connection_string='{connection_string}';")

        all_players_data = {}
        
        for jersey_number in jersey_numbers:
            print(f"\nReading player #{jersey_number}...")
            
            # Query the specific player file
            query = f"""
                SELECT *
                FROM read_parquet('azure://{CONTAINER_NAME}/teams/team_{team_id}/{jersey_number:02d}.parquet')
            """
            
            # Execute query and convert to DataFrame
            df = con.execute(query).fetchdf()
            
            # Process the data for better display
            # Handle allocations
            allocation_columns = [
                'defensive_position_allocation_one',
                'defensive_position_allocation_two',
                'defensive_position_allocation_three',
                'defensive_position_allocation_four'
            ]
            for col in allocation_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: f"{float(x):.2f}" if pd.notnull(x) else None
                    )

            # Handle positions
            position_columns = [
                'defensive_position_one',
                'defensive_position_two',
                'defensive_position_three',
                'defensive_position_four'
            ]
            for col in position_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).where(pd.notnull(df[col]), None)
            
            print(f"\nPlayer #{jersey_number} Data:")
            print("-" * 50)
            
            # Print each column's data
            for column in df.columns:
                value = df[column].iloc[0]
                print(f"{column}: {value}")
            
            # Save player data to dictionary
            player_dict = df.to_dict(orient='records')[0]
            all_players_data[jersey_number] = player_dict
        
        # Save all data to JSON for reference
        import json
        with open('players_data.json', 'w') as f:
            json.dump(all_players_data, f, indent=4)
        print(f"\nAll data saved to 'players_data.json'")
        
        return all_players_data
        
    except Exception as e:
        print(f"Error reading player parquet: {str(e)}")
        raise

if __name__ == "__main__":
    print("Reading players data from team 3...")
    players_data = read_player_parquet("3", [9, 22])
    print("\nRead completed successfully!") 