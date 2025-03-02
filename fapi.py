from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import os
from utils import *

# Import routers instead of apps
from teams import router as teams_router
from games import router as games_router
from lineup import router as lineup_router

app = FastAPI(
    title="FastAPI Team Management API",
    description="API for managing teams, players, and games",
    version="1.0.0"
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Include routers instead of mounting apps
app.include_router(
    teams_router,
    prefix="/teams",
    tags=["teams"]
)

app.include_router(
    games_router,
    prefix="/games",
    tags=["games"]
)

app.include_router(
    lineup_router,
    prefix="/lineup",
    tags=["lineup"]
)

@app.get("/", tags=["root"])
def read_root():
    return {"message": "FastAPI Team Management API"}

# Add favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")