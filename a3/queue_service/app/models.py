from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QueueRole(str, Enum):
    """Roles for queue access control"""
    ADMIN = "admin"   # full access to create, delete, modify queues
    AGENT = "agent"   # they can only push and pull from the queues
    USER = "user"     # user doesnt have any permission


class MessageBase(BaseModel):
    """Base model for queue messages
    
    1. Transaction data format from our last assignment:
       {
         "transaction_id": str,
         "customer_id": str,
         "customer_name": str,
         "amount": float,
         "vendor_id": str,
         "date": str,
         ... other transaction fields
       }
    
    2. Prediction result format from Assignment 2:
       {
         "transaction_id": str,
         "prediction": bool,  # True for approved, False for rejected
         "confidence": float, # Confidence score of the prediction
         "model_version": str,
         "timestamp": str,
         ... other prediction fields
       }
    """
    content: Dict[str, Any]  # Content can be any valid json object (transaction or prediction)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str = "transaction"  # Can be "transaction" or "prediction"


class Message(MessageBase):
    """Message model with unique ID"""
    id: str  # Unique message ID


class QueueType(str, Enum):    # our intention is to have queues of 2 types (either transaction or prediction queue)
    """Types of queues"""
    TRANSACTION = "transaction"
    PREDICTION = "prediction"


class QueueConfig(BaseModel):
    """Queue configuration model"""
    max_messages: int = 5    # Maximum number of messages allowed in the queue
    persist_interval_seconds: int = 60   # how long till we save queue state to disk
    queue_type: QueueType = QueueType.TRANSACTION


class QueueInfo(BaseModel):
    """Queue information model"""
    name: str
    message_count: int      # this is number of current messages in queue
    queue_type: QueueType   # type of queue (transaction/prediction)
    created_at: datetime    # queue creation time
    last_modified: datetime = Field(default_factory=datetime.utcnow)  # time last modified


class QueueCreate(BaseModel):
    """Model for queue creation"""
    name: str
    config: Optional[QueueConfig] = None


class QueueList(BaseModel):
    """Model for listing queues"""
    queues: List[QueueInfo]


class ErrorResponse(BaseModel):
    """Standard error response model"""
    detail: str


class TokenData(BaseModel):
    """Model for token data"""
    username: str
    role: QueueRole   # User's role, important for permissions
