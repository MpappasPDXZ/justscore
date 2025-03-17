# parquet_reader.py
import os
import pandas as pd
import numpy as np
import json
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def read_parquet_from_azure():
    """
    Prompts the user for an inning number (e.g., 1) and a batter sequence ID (e.g., 7),
    and then reads that specific parquet file from Azure.
    """
    # Ask for inning and batter seq
    inning_number = input("Enter the inning number (e.g., 1): ")
    batter_seq_id = input("Enter the batter sequence ID (e.g., 7): ")

    # Hardcode or retrieve the team/game:
    team_id = "1"
    game_id = "1"

    # Construct blob_path using user inputs
    blob_path = f"games/team_{team_id}/game_{game_id}/inning_{inning_number}/away_{batter_seq_id}.parquet"
    
    # Get connection string from environment variable
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        return
    
    # Set up the blob client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "justscorecontainer"
    
    try:
        # Get the blob client
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Download the blob
        print(f"Downloading {blob_path}...")
        blob_data = blob_client.download_blob().readall()
        
        # Read the Parquet data
        buffer = BytesIO(blob_data)
        df = pd.read_parquet(buffer)
        
        # Print the entire DataFrame for inspection
        print("\nDataFrame Contents:")
        print(df)
        
        # Print just the column list for SQL without quotes
        print("\nSQL Column List (without quotes):")
        for i, col in enumerate(df.columns):
            comma = "," if i < len(df.columns) - 1 else ""
            print(f'{col}{comma}')
        
        # Examine hit_to field
        if 'hit_to' in df.columns:
            value = df['hit_to'].iloc[0]
            print("\n=== hit_to Analysis ===")
            print(f"Type: {type(value)}")
            print(f"Value: {value}")
            
            # Try to understand if it's a scalar or array
            if isinstance(value, (list, np.ndarray, pd.Series)):
                print("hit_to appears to be an array type")
                if hasattr(value, 'tolist'):
                    list_value = value.tolist()
                    print(f"As list: {list_value}")
            elif isinstance(value, str):
                print("hit_to is a string type")
                try:
                    # Try to parse as JSON in case it's a serialized array
                    json_value = json.loads(value)
                    print(f"As parsed JSON: {json_value}")
                except:
                    print("Not a JSON string")
            else:
                print(f"hit_to is a scalar of type {type(value)}")
        else:
            print("\nWARNING: 'hit_to' column not found in the Parquet file")
        
        # Focus on hit_around_bases field
        if 'hit_around_bases' in df.columns:
            value = df['hit_around_bases'].iloc[0]
            print("\n=== hit_around_bases Analysis ===")
            print(f"Type: {type(value)}")
            print(f"Value: {value}")
            
            # Try different methods to convert to list
            print("\nConversion attempts:")
            
            # Method 1: tolist() if available
            if hasattr(value, 'tolist'):
                list_value = value.tolist()
                print(f"tolist() result: {list_value} (type: {type(list_value)})")
            
            # Method 2: list() conversion
            try:
                list_value = list(value)
                print(f"list() result: {list_value} (type: {type(list_value)})")
            except Exception as e:
                print(f"list() conversion failed: {str(e)}")
            
            # Method 3: JSON parsing if it's a string
            if isinstance(value, str):
                try:
                    json_value = json.loads(value)
                    print(f"json.loads() result: {json_value} (type: {type(json_value)})")
                except Exception as e:
                    print(f"JSON parsing failed: {str(e)}")
            
            # Save the raw value to a file for inspection
            with open('hit_around_bases_raw.txt', 'w') as f:
                f.write(str(value))
                f.write("\n\nType: " + str(type(value)))
            
            print("\nRaw value saved to hit_around_bases_raw.txt for inspection")
        else:
            print("\nWARNING: 'hit_around_bases' column not found in the Parquet file")
            print("Available columns:", df.columns.tolist())
        
        # Check for stolen_bases field too
        if 'stolen_bases' in df.columns:
            value = df['stolen_bases'].iloc[0]
            print("\n=== stolen_bases Analysis ===")
            print(f"Type: {type(value)}")
            print(f"Value: {value}")
        
        return df
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    df = read_parquet_from_azure()