from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Literal, Optional, List, Dict
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import duckdb
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
import json
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure Storage configuration
ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT')
ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
CONTAINER_NAME = "justscorecontainer"

# Utility function to get blob service client
def get_blob_service_client():
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def get_duckdb_connection():
    """
    Create and configure a DuckDB connection for Azure Blob Storage access
    """
    con = duckdb.connect()
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    con.execute("SET azure_transport_option_type = 'curl';")
    con.execute(f"SET azure_storage_connection_string='{connection_string}';")
    return con

def blob_exists(blob_client):
    try:
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False
    
def count_errors(error_array):
    """Count elements in a NumPy array"""
    if error_array is None:
        return 0       
    try:
        if isinstance(error_array, np.ndarray):
            if error_array.size == 0:
                return 0
            return len(error_array)
        return 0
    except Exception as e:
        print(f"Error counting: {str(e)}")
        return 0

def process_array_field(value):
    """Helper function to process array-type fields"""
    # Convert to list if it's not already
    if isinstance(value, (list, pd.Series, np.ndarray)):
        if hasattr(value, 'tolist'):
            return value.tolist()
        else:
            return list(value)
    elif isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            if value and not pd.isna(value):  
                return [value]
            else:
                return []
    else:
        return []