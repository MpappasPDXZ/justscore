import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_get_team_roster(team_id: str):
    try:
        # Define the API endpoint
        api_url = f"http://localhost:8000/teams/{team_id}/roster"
        
        # Make the GET request
        print(f"\nFetching roster for team {team_id}...")
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create timestamp for the filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"team_{team_id}_roster_{timestamp}.json"
            
            # Save the raw response
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"\nRaw response saved to: {filename}")
            
            # Print roster details
            print(f"\nTeam {team_id} Roster:")
            print("-" * 50)
            
            if data.get("roster"):
                for player in data["roster"]:
                    print(f"\nPlayer: {player.get('player_name')}")
                    print(f"Jersey: {player.get('jersey_number')}")
                    print(f"Status: {player.get('active')}")
                    print(f"Primary Position: {player.get('defensive_position_one')} ({player.get('defensive_position_allocation_one')})")
                    if player.get('defensive_position_two'):
                        print(f"Secondary Position: {player.get('defensive_position_two')} ({player.get('defensive_position_allocation_two')})")
                    if player.get('defensive_position_three'):
                        print(f"Third Position: {player.get('defensive_position_three')} ({player.get('defensive_position_allocation_three')})")
                    if player.get('defensive_position_four'):
                        print(f"Fourth Position: {player.get('defensive_position_four')} ({player.get('defensive_position_allocation_four')})")
                    print("-" * 30)
                
                print(f"\nTotal players: {len(data['roster'])}")
            else:
                print("\nNo players found in roster")
            
            return data
            
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error testing roster endpoint: {str(e)}")
        return None

if __name__ == "__main__":
    # Test roster for team 3
    print("Starting roster test...")
    result = test_get_team_roster("3")
    if result:
        print("\nRoster test completed successfully!")
    else:
        print("\nRoster test failed!") 