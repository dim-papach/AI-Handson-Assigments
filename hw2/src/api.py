import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure root directory is in path to resolve hw2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hw2.src.agent import invoke_agent, stream_agent

app = FastAPI(
    title="HECATE Astrophysics Agent API",
    description="Exposes the HECATE RAG and classification agent via a REST interface.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The message/query to send to the astrophysics agent.")
    session_id: str = Field(..., description="Unique thread/session ID to maintain conversation memory.")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's text response.")

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """
    Exposes the LangGraph conversational agent.
    Maintains conversation memory across turns within a session (session_id).
    """
    try:
        # Check API key presence dynamically
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            raise HTTPException(
                status_code=500,
                detail="Google/Gemini API key is not configured on the server."
            )
            
        result = invoke_agent(payload.message, payload.session_id)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    """
    Streams the agent's response token-by-token using Server-Sent Events (SSE).
    """
    try:
        # Check API key presence dynamically
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            raise HTTPException(
                status_code=500,
                detail="Google/Gemini API key is not configured on the server."
            )
            
        return StreamingResponse(
            stream_agent(payload.message, payload.session_id),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}
