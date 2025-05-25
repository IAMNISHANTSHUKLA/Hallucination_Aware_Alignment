# backend/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from typing import Dict, List
import json

from config import settings
from api.haa_routes import router as haa_router
from api.agent_routes import router as agent_router
from api.evaluation_routes import router as evaluation_router
from core.multi_agent_system import MultiAgentSystem
from utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hallucination-Aware Alignment (HAA) System",
    description="Research-grade LLM alignment system with hallucination detection and suppression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global HAA system
haa_system = None
active_connections: Dict[str, WebSocket] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize HAA system on startup"""
    global haa_system
    try:
        logger.info("Initializing HAA System...")
        haa_system = MultiAgentSystem(config=settings)
        await haa_system.initialize()
        logger.info("HAA System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize HAA System: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global haa_system
    if haa_system:
        await haa_system.cleanup()
    logger.info("HAA System shutdown complete")

# Include routers
app.include_router(haa_router, prefix="/api/v1/haa", tags=["HAA Core"])
app.include_router(agent_router, prefix="/api/v1/agents", tags=["Multi-Agent System"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["Evaluation"])

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "subscribe_metrics":
                # Start sending metrics updates
                asyncio.create_task(send_metrics_updates(client_id))
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

async def send_metrics_updates(client_id: str):
    """Send periodic metrics updates to connected clients"""
    while client_id in active_connections:
        try:
            if haa_system:
                metrics = await haa_system.get_system_metrics()
                await active_connections[client_id].send_text(
                    json.dumps({"type": "metrics", "data": metrics})
                )
        except Exception as e:
            logger.error(f"Error sending metrics to {client_id}: {e}")
            break
        
        await asyncio.sleep(1)  # Send updates every second

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if haa_system and haa_system.is_ready() else "unhealthy"
    return {
        "status": status,
        "version": "1.0.0",
        "system_ready": haa_system.is_ready() if haa_system else False
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Hallucination-Aware Alignment (HAA) System",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )