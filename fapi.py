from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.encoders import jsonable_encoder
import os
from utils import *

# Import routers instead of apps
from games import router as games_router
from teams import router as teams_router
from lineup import router as lineup_router
from scores import router as score_router
from defense import router as defense_router

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
    allow_headers=["*"],
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

app.include_router(
    score_router,
    prefix="/scores",
    tags=["scores"]
)

app.include_router(
    defense_router,
    prefix="/defense",
    tags=["defense"]
)

@app.get("/", tags=["root"])
def read_root():
    return {"message": "FastAPI Team Management API v7.0.1"}

# Add favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")