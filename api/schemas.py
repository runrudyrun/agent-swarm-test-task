"""Pydantic schemas for API requests and responses."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's query message")
    user_id: Optional[str] = Field(None, description="User ID for personalized responses")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    answer: str = Field(..., description="Agent's response to the query")
    agent_used: str = Field(..., description="Agent that processed the query")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: float = Field(0.0, description="Confidence score of the response")
    sources: Optional[List[str]] = Field(None, description="Source URLs for knowledge responses")
    handoff_to_human: bool = Field(False, description="Whether query should be escalated to human")
    requires_user_id: bool = Field(False, description="Whether user ID is required for this query")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    agents: Dict[str, str] = Field(..., description="Status of individual agents")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")