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