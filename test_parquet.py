from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_depth_chart_parquet():
    try:
        # Get Azure credentials from environment variables
        ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
        ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        
        # Set up the blob service client
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get container client
        container_client = blob_service_client.get_container_client("justscorecontainer")
        
        # Get blob client
        blob_path = "teams/team_1/depth_chart/depth_chart.parquet"
        blob_client = container_client.get_blob_client(blob_path)
        
        # Download blob data
        blob_data = blob_client.download_blob().readall()
        
        # Convert to DataFrame
        df = pd.read_parquet(BytesIO(blob_data))
        
        # Print the data
        print("\nDataFrame contents:")
        print(df)
        
        # Print some basic info
        print("\nDataFrame info:")
        print(df.info())
        
        # Print unique positions
        print("\nUnique positions:")
        print(df['position'].unique())
        
        return df
        
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting parquet file read test...")
    df = read_depth_chart_parquet()
    print("\nTest completed successfully!") 