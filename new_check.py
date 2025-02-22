import requests
import json
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

def get_all_teams_metadata():
    try:
        # Define the API endpoint
        api_url = "http://localhost:8000/read_metadata_duckdb"
        
        # Make the GET request
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metadata to a single JSON file
            filename = f"teams_metadata_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"\nTeam metadata exported to: {filename}")
            print(f"Total teams exported: {data.get('total_teams', 0)}")
            
            return data
            
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting teams metadata: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting metadata export...")
    result = get_all_teams_metadata()
    if result:
        print("Metadata export completed successfully!")
    else:
        print("Metadata export failed!")
