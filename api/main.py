"""FastAPI application for Agent Swarm system."""

import logging
import os
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.personality import PersonalityLayer
from agents.router_agent import RouterAgent
from api.schemas import ErrorResponse, HealthResponse, QueryRequest, QueryResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="InfinitePay Agent Swarm API",
    description="Multi-agent system for customer support and knowledge retrieval",
    version="0.1.0",
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

# Initialize agents
router_agent = RouterAgent()
personality_layer = PersonalityLayer()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "InfinitePay Agent Swarm API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    timestamp = datetime.now().isoformat()
    
    # Check agent statuses
    agents_status = {
        "router": "healthy",
        "knowledge": "healthy" if router_agent.knowledge_agent.is_available() else "unavailable",
        "support": "healthy",
        "personality": "healthy" if personality_layer.is_enabled() else "disabled"
    }
    
    # Overall status
    overall_status = "healthy"
    if any(status == "unavailable" for status in agents_status.values()):
        overall_status = "degraded"
    elif any(status == "unhealthy" for status in agents_status.values()):
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        version="0.1.0",
        agents=agents_status
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main query endpoint."""
    try:
        logger.info(f"Received query: {request.message[:100]}...")
        
        # Route query to appropriate agent
        result = router_agent.route_query(request.message, request.user_id)
        result["lang"] = result.get("lang", "pt")
        
        # Apply personality layer if enabled
        if personality_layer.is_enabled():
            lang = result.get("lang", "pt")
            result["answer"] = personality_layer.adjust_response(
                result["answer"],
                context=result,
                lang=lang
            )
        
        # Create response
        response = QueryResponse(
            answer=result["answer"],
            agent_used=result["agent_used"],
            intent=result.get("intent"),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources"),
            handoff_to_human=result.get("handoff_to_human", False),
            requires_user_id=result.get("requires_user_id", False)
        )
        
        logger.info(f"Query processed by {result['agent_used']} with confidence {result.get('confidence', 0.0)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Please try again later."
        )


@app.get("/capabilities")
async def get_capabilities():
    """Get capabilities of available agents."""
    return {
        "agents": router_agent.get_agent_capabilities(),
        "features": {
            "multi_intent": True,
            "personality_layer": personality_layer.is_enabled(),
            "human_escalation": True,
            "source_citations": True
        },
        "supported_locales": ["pt-BR"],
        "max_query_length": 1000
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting InfinitePay Agent Swarm API on port {port}")
    logger.info(f"Knowledge agent available: {router_agent.knowledge_agent.is_available()}")
    logger.info(f"Personality layer enabled: {personality_layer.is_enabled()}")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )