from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from starlette.responses import StreamingResponse
import uvicorn
import json
import time
import signal
import sys

from .models import (
    QueueInfo, QueueCreate, QueueList, Message, MessageBase, 
    TokenData, QueueRole, ErrorResponse
)
from .queue_manager import queue_manager
from .auth import (
    get_current_user, validate_admin_privileges, 
    validate_agent_or_admin_privileges, create_access_token
)
from .logger import logger, log_request_response
from .config import config


# Lifespan context manager (modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Queue Service starting up...")
    # Log configuration
    logger.info(f"Using configuration: {config.get_all()}")
    
    yield
    
    # Shutdown
    logger.info("Queue Service shutting down...")
    queue_manager.persist_all()

# Create FastAPI application
app = FastAPI(
    title="Queue Service",
    description="Message queue service for high performance computing tasks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware for response logging
class ResponseLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate a request ID
        request_id = str(time.time())
        
        # Get client IP
        client_host = request.client.host if request.client else "unknown"
        
        # Capture request body
        try:
            # Store original request body
            body_bytes = await request.body()
            # Create a new stream with the same content
            request._body = body_bytes
            
            # Try to parse body as JSON or form data
            body_str = body_bytes.decode('utf-8') if body_bytes else ""
            if body_str and body_str.strip().startswith('{'):
                try:
                    request_body = json.loads(body_str)
                except:
                    request_body = body_str
            elif body_str and '=' in body_str:  # Form data
                try:
                    request_body = {k: v for k, v in [item.split('=') for item in body_str.split('&')]}
                except:
                    request_body = body_str
            else:
                request_body = body_str
        except Exception as e:
            request_body = {"error": f"Could not read request body: {str(e)}"}
        
        # Log request
        from .logger import log_request_response
        log_request_response(
            source=client_host,
            destination=request.url.path,
            headers=dict(request.headers),
            metadata={
                "method": request.method,
                "request_id": request_id
            },
            body=request_body
        )
        
        # Process the request
        start_time = time.time()
        
        # Create a custom response class that captures the body
        original_response = await call_next(request)
        process_time = time.time() - start_time
        
        # Create a modified response with logging
        response_body = await self._get_response_body(original_response)
        
        # Log response with the body
        log_request_response(
            source=request.url.path,
            destination=client_host,
            headers=dict(original_response.headers),
            metadata={
                "status_code": original_response.status_code,
                "process_time_ms": round(process_time * 1000),
                "request_id": request_id
            },
            body=response_body
        )
        
        # Return the original response
        return original_response
    
    async def _get_response_body(self, response):
        """Extract the body from a response"""
        if not hasattr(response, "body"):
            return {}
            
        try:
            # Read the response body
            body = response.body
            
            # Try to parse as JSON
            try:
                return json.loads(body)
            except:
                # Return as string if not JSON
                return body.decode('utf-8')
        except Exception as e:
            logger.error(f"Error capturing response body: {str(e)}")
            return {}

# Add the middleware to the app
app.add_middleware(ResponseLoggingMiddleware)


# Graceful shutdown handler
def handle_shutdown(sig, frame):
    """Handle shutdown signals and ensure queue data is persisted"""
    logger.info("Received shutdown signal, persisting queue data...")
    queue_manager.persist_all()
    logger.info("Queue data persisted, shutting down...")
    # Don't call sys.exit() as it triggers SystemExit exceptions
    # Let the application terminate naturally


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Web UI
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def get_ui(request: Request):
    """Serve the web UI"""
    return templates.TemplateResponse("index.html", {"request": request})

# Current user endpoint for UI
@app.get("/current-user", tags=["Authentication"])
async def get_current_user(token_data: TokenData = Depends(get_current_user)):
    """Get information about the current user"""
    return {"username": token_data.username, "role": token_data.role}


# Authentication endpoints
@app.post("/token", tags=["Authentication"])
async def login_for_access_token(username: str, password: str):
    """
    Get an access token for authentication
    
    This is a simplified auth endpoint for demonstration purposes
    In a production environment, you would validate credentials against a database
    """
    # For demo purposes, we'll use hardcoded credentials
    # In a real application, this would validate against a database
    valid_users = {
        "admin": {"password": "admin_password", "role": QueueRole.ADMIN},
        "agent": {"password": "agent_password", "role": QueueRole.AGENT},
        "user": {"password": "user_password", "role": QueueRole.USER}
    }
    
    # Check if user exists and password is correct
    if username not in valid_users or valid_users[username]["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    token_data = {
        "sub": username,
        "role": valid_users[username]["role"]
    }
    access_token = create_access_token(token_data)
    
    return {"access_token": access_token, "token_type": "bearer"}


# Queue management endpoints
@app.get("/queues", response_model=QueueList, tags=["Queue Management"])
async def list_queues(token_data: TokenData = Depends(get_current_user)):
    """
    List all queues
    
    Returns:
        List of queues
    """
    queues = queue_manager.list_queues()
    return QueueList(queues=queues)


@app.post("/queues", response_model=QueueInfo, tags=["Queue Management"])
async def create_queue(
    queue_data: QueueCreate,
    token_data: TokenData = Depends(validate_admin_privileges)
):
    """
    Create a new queue
    
    Only administrators can create queues
    Queue type must be specified as either 'transaction' or 'prediction'
    
    Args:
        queue_data: Queue creation data with queue_type
        
    Returns:
        Created queue information
    """
    # Ensure queue config has a queue_type
    if not queue_data.config or not hasattr(queue_data.config, 'queue_type'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Queue type must be specified (transaction or prediction)"
        )
    
    success, message = await queue_manager.create_queue(
        queue_data.name,
        queue_data.config
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # Get the queue info
    queue_info = await queue_manager.get_queue_info(queue_data.name)
    if not queue_info:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get queue info after creation"
        )
    
    return queue_info


@app.delete("/queues/{queue_name}", tags=["Queue Management"])
async def delete_queue(
    queue_name: str,
    token_data: TokenData = Depends(validate_admin_privileges)
):
    """
    Delete a queue
    
    Only administrators can delete queues
    All messages in the queue will be deleted
    
    Args:
        queue_name: Name of the queue to delete
        
    Returns:
        Success message
    """
    success, message = await queue_manager.delete_queue(queue_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
    return {"message": message}


@app.get("/queues/{queue_name}", response_model=QueueInfo, tags=["Queue Management"])
async def get_queue_info(
    queue_name: str,
    token_data: TokenData = Depends(get_current_user)
):
    """
    Get information about a queue
    
    Args:
        queue_name: Name of the queue
        
    Returns:
        Queue information
    """
    queue_info = await queue_manager.get_queue_info(queue_name)
    
    if not queue_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue '{queue_name}' not found"
        )
    
    return queue_info


# Message operations endpoints
@app.post("/queues/{queue_name}/push", tags=["Message Operations"])
async def push_message(
    queue_name: str,
    message: Dict[str, Any],
    message_type: str = "transaction",  # Can be "transaction" or "prediction"
    token_data: TokenData = Depends(validate_agent_or_admin_privileges)
):
    """
    Push a message to a queue
    
    Only agents and administrators can push messages
    Agents can only push prediction messages, not transaction messages
    Messages must match the queue type (transaction or prediction)
    
    Args:
        queue_name: Name of the queue
        message: Message content (transaction data or prediction result from Assignment 2)
        message_type: Type of message - either "transaction" or "prediction"
        
    Returns:
        Success message and message ID
    """
    success, message_text, message_id = await queue_manager.push_message(
        queue_name=queue_name, 
        content=message, 
        message_type=message_type,
        user_role=token_data.role
    )
    
    if not success:
        if "does not exist" in message_text:
            status_code = status.HTTP_404_NOT_FOUND
        elif "is full" in message_text:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        
        raise HTTPException(
            status_code=status_code,
            detail=message_text
        )
    
    return {"message": message_text, "message_id": message_id}


@app.get("/queues/{queue_name}/pull", tags=["Message Operations"])
async def pull_message(
    queue_name: str,
    token_data: TokenData = Depends(validate_agent_or_admin_privileges),
    request: Request = None
):
    """
    Pull a message from a queue
    
    Only agents and administrators can pull messages
    Removes the first (oldest) message from the queue and returns its contents
    
    Args:
        queue_name: Name of the queue
        
    Returns:
        Message content
    """
    success, message_text, pulled_message = await queue_manager.pull_message(queue_name)
    
    if not success:
        if "does not exist" in message_text:
            status_code = status.HTTP_404_NOT_FOUND
        elif "is empty" in message_text:
            status_code = status.HTTP_204_NO_CONTENT
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        
        raise HTTPException(
            status_code=status_code,
            detail=message_text
        )
    
    # Prepare the response body
    response_body = {
        "message_id": pulled_message.id,
        "content": pulled_message.content,
        "timestamp": pulled_message.timestamp,
        "message_type": pulled_message.message_type
    }
    
    return response_body


# Custom exception handler for structured error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# Note: The startup and shutdown events are now handled by the lifespan context manager above


# Run the server if executed directly
if __name__ == "__main__":
    port = config.get("port", 7500)
    host = config.get("host", "localhost")
    
    logger.info(f"Starting Queue Service on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
