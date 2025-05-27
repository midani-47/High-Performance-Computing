import json
import os
import uuid
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
from collections import deque

from .models import Message, QueueInfo, QueueConfig
from .logger import logger
from .config import config


class QueueManager:
    """
    Manager for all message queues
    
    Handles:
    - Queue creation and deletion
    - Message push and pull operations
    - Persistent storage of queue state
    - Thread-safe queue operations
    """
    
    def __init__(self):
        # Initialize state
        self._queues: Dict[str, deque] = {}  # Queue name -> deque of messages
        self._queue_info: Dict[str, QueueInfo] = {}  # Queue name -> queue metadata
        self._queue_configs: Dict[str, QueueConfig] = {}  # Queue name -> queue config
        self._locks: Dict[str, asyncio.Lock] = {}  # Queue name -> lock
        
        # Setup persistence
        self._storage_path = Path(config.get("storage_path", "./queue_data"))
        self._storage_path.mkdir(exist_ok=True, parents=True)
        
        # Load existing queues from storage
        self._load_queues()
        
        # Set up persistence thread
        self._last_persist_time = time.time()
        self._persist_interval = config.get("persist_interval_seconds", 60)
        self._persist_thread = threading.Thread(target=self._persistence_worker, daemon=True)
        self._persist_thread.start()
    
    def _persistence_worker(self):
        """Background thread for periodically persisting queue state"""
        while True:
            time.sleep(1)  # Check every second
            current_time = time.time()
            
            # Check if it's time to persist
            if current_time - self._last_persist_time >= self._persist_interval:
                self.persist_all()
                self._last_persist_time = current_time
    
    def _load_queues(self):
        """Load all queues from persistent storage"""
        if not self._storage_path.exists():
            logger.info("Storage path does not exist, creating...")
            self._storage_path.mkdir(exist_ok=True, parents=True)
            return
        
        # Load queue metadata
        metadata_path = self._storage_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Recreate queues from metadata
                for queue_name, queue_data in metadata.items():
                    # Handle existing queues that don't have queue_type
                    # Default to transaction type for backward compatibility
                    queue_type = queue_data.get("queue_type", "transaction")
                    if isinstance(queue_type, str):
                        queue_type = queue_type
                    else:
                        queue_type = "transaction"
                        
                    self._queue_info[queue_name] = QueueInfo(
                        name=queue_name,
                        message_count=queue_data.get("message_count", 0),
                        queue_type=queue_type,
                        created_at=datetime.fromisoformat(queue_data.get("created_at")),
                        last_modified=datetime.fromisoformat(queue_data.get("last_modified"))
                    )
                    
                    self._queue_configs[queue_name] = QueueConfig(
                        max_messages=queue_data.get("max_messages", 1000),
                        persist_interval_seconds=queue_data.get("persist_interval_seconds", 60)
                    )
                    
                    # Initialize the queue
                    self._queues[queue_name] = deque()
                    self._locks[queue_name] = asyncio.Lock()
                    
                    # Load queue messages
                    queue_path = self._storage_path / f"{queue_name}.json"
                    if queue_path.exists():
                        try:
                            with open(queue_path, "r") as f:
                                messages = json.load(f)
                                for msg in messages:
                                    self._queues[queue_name].append(
                                        Message(
                                            id=msg.get("id"),
                                            content=msg.get("content"),
                                            timestamp=datetime.fromisoformat(msg.get("timestamp"))
                                        )
                                    )
                            logger.info(f"Loaded {len(self._queues[queue_name])} messages for queue {queue_name}")
                        except Exception as e:
                            logger.error(f"Error loading messages for queue {queue_name}: {str(e)}")
                
                logger.info(f"Loaded {len(self._queues)} queues from persistent storage")
            except Exception as e:
                logger.error(f"Error loading queue metadata: {str(e)}")
    
    def persist_all(self):
        """Persist all queues to storage"""
        if not self._queues:
            return
        
        # Ensure storage path exists
        self._storage_path.mkdir(exist_ok=True, parents=True)
        
        # Persist queue metadata
        metadata = {}
        for queue_name, info in self._queue_info.items():
            config = self._queue_configs.get(queue_name, QueueConfig())
            metadata[queue_name] = {
                "message_count": len(self._queues[queue_name]),
                "created_at": info.created_at.isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "max_messages": config.max_messages,
                "persist_interval_seconds": config.persist_interval_seconds,
                "queue_type": info.queue_type
            }
        
        try:
            with open(self._storage_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting queue metadata: {str(e)}")
        
        # Persist each queue's messages
        for queue_name, queue in self._queues.items():
            try:
                messages = []
                for msg in queue:
                    messages.append({
                        "id": msg.id,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    })
                
                with open(self._storage_path / f"{queue_name}.json", "w") as f:
                    json.dump(messages, f, indent=2)
                
                logger.info(f"Persisted {len(messages)} messages for queue {queue_name}")
            except Exception as e:
                logger.error(f"Error persisting messages for queue {queue_name}: {str(e)}")
    
    async def create_queue(self, name: str, config: Optional[QueueConfig] = None) -> Tuple[bool, str]:
        """
        Create a new queue
        
        Args:
            name: Name of the queue to create
            config: Optional queue configuration with queue_type (transaction or prediction)
            
        Returns:
            (success, message)
        """
        # Validate queue name
        if not name or not name.isalnum():
            return False, "Queue name must be alphanumeric"
        
        # Check if queue already exists
        if name in self._queues:
            return False, f"Queue with name '{name}' already exists"
        
        # Create the queue
        self._queues[name] = deque()
        self._locks[name] = asyncio.Lock()
        
        # Set configuration
        if config:
            self._queue_configs[name] = config
        else:
            # Use the max_messages_per_queue from the config file
            max_messages = config.get("max_messages_per_queue", 5)
            self._queue_configs[name] = QueueConfig(max_messages=max_messages)
        
        # Create queue info with the queue type
        self._queue_info[name] = QueueInfo(
            name=name,
            message_count=0,
            queue_type=self._queue_configs[name].queue_type,
            created_at=datetime.utcnow(),
            last_modified=datetime.utcnow()
        )
        
        logger.info(f"Created queue '{name}'")
        return True, f"Queue '{name}' created successfully"
    
    async def delete_queue(self, name: str) -> Tuple[bool, str]:
        """
        Delete a queue
        
        Args:
            name: Name of the queue to delete
            
        Returns:
            (success, message)
        """
        # Check if queue exists
        if name not in self._queues:
            return False, f"Queue '{name}' does not exist"
        
        # Delete the queue
        async with self._locks[name]:
            del self._queues[name]
            del self._queue_info[name]
            del self._queue_configs[name]
            
            # Delete the queue's persistent storage
            queue_path = self._storage_path / f"{name}.json"
            if queue_path.exists():
                try:
                    os.remove(queue_path)
                except Exception as e:
                    logger.error(f"Error deleting queue file: {str(e)}")
        
        # Remove the lock after it's no longer needed
        del self._locks[name]
        
        logger.info(f"Deleted queue '{name}'")
        return True, f"Queue '{name}' deleted successfully"
    
    def list_queues(self) -> List[QueueInfo]:
        """
        List all queues
        
        Returns:
            List of queue information objects
        """
        return list(self._queue_info.values())
    
    async def push_message(self, queue_name: str, content: Dict[str, Any], message_type: str = "transaction", user_role: str = "admin") -> Tuple[bool, str, Optional[str]]:
        """
        Push a message to the queue
        
        Args:
            queue_name: Name of the queue
            content: Message content (either transaction data or prediction result from Assignment 2)
            message_type: Type of message - either "transaction" or "prediction"
            user_role: Role of the user making the request (admin or agent)
            
        Returns:
            (success, message, message_id)
        """
        # Check if queue exists
        if queue_name not in self._queues:
            return False, f"Queue '{queue_name}' does not exist", None
        
        # Validate message type
        if message_type not in ["transaction", "prediction"]:
            return False, f"Invalid message type: {message_type}. Must be 'transaction' or 'prediction'", None
        
        # Get queue type
        queue_type = self._queue_info[queue_name].queue_type
        
        # Check if message type is compatible with queue type
        if message_type == "transaction" and queue_type != "transaction":
            return False, f"Cannot push transaction message to prediction queue '{queue_name}'", None
        
        if message_type == "prediction" and queue_type != "prediction":
            return False, f"Cannot push prediction message to transaction queue '{queue_name}'", None
        
        # Both admins and agents can push any message type
        # No need to check role permissions here as this is already handled by the endpoint
        
        # Validate message content based on type
        if message_type == "transaction":
            required_fields = ["transaction_id", "customer_id", "amount", "vendor_id"]
            for field in required_fields:
                if field not in content:
                    return False, f"Transaction message missing required field: {field}", None
        elif message_type == "prediction":
            required_fields = ["transaction_id", "prediction", "confidence"]
            for field in required_fields:
                if field not in content:
                    return False, f"Prediction message missing required field: {field}", None
        
        async with self._locks[queue_name]:
            # Check queue size limit
            max_messages = self._queue_configs[queue_name].max_messages
            if len(self._queues[queue_name]) >= max_messages:
                return False, f"Queue '{queue_name}' is full (max {max_messages} messages)", None
            
            # Create and add the message
            message_id = str(uuid.uuid4())
            message = Message(
                id=message_id, 
                content=content, 
                timestamp=datetime.utcnow(),
                message_type=message_type
            )
            
            self._queues[queue_name].append(message)
            
            # Update queue metadata
            self._queue_info[queue_name].message_count = len(self._queues[queue_name])
            self._queue_info[queue_name].last_modified = datetime.utcnow()
        
        # Log the message push operation with full body content
        from .logger import log_message
        log_message(
            queue_name=queue_name,
            message={
                "id": message_id,
                "message_type": message_type,
                "content": content,
                "body": content  # Explicitly include body for logging
            },
            action="push"
        )
        
        logger.info(f"Pushed message {message_id} to queue '{queue_name}'")
        return True, f"Message pushed to queue '{queue_name}'", message_id
    
    async def pull_message(self, queue_name: str) -> Tuple[bool, str, Optional[Message]]:
        """
        Pull a message from the queue
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            (success, message, pulled_message)
        """
        # Check if queue exists
        if queue_name not in self._queues:
            return False, f"Queue '{queue_name}' does not exist", None
        
        async with self._locks[queue_name]:
            # Check if queue is empty
            if not self._queues[queue_name]:
                return False, f"Queue '{queue_name}' is empty", None
            
            # Get the message from the front of the queue
            message = self._queues[queue_name].popleft()
            
            # Update queue metadata
            self._queue_info[queue_name].message_count = len(self._queues[queue_name])
            self._queue_info[queue_name].last_modified = datetime.utcnow()
        
        # Log the message pull operation with full body content
        from .logger import log_message
        log_message(
            queue_name=queue_name,
            message={
                "id": message.id,
                "message_type": message.message_type,
                "content": message.content,
                "body": message.content  # Explicitly include body for logging
            },
            action="pull"
        )
        
        logger.info(f"Pulled message {message.id} from queue '{queue_name}'")
        return True, f"Message pulled from queue '{queue_name}'", message
    
    async def get_queue_info(self, queue_name: str) -> Optional[QueueInfo]:
        """
        Get information about a queue
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue information or None if queue doesn't exist
        """
        if queue_name not in self._queue_info:
            return None
        
        return self._queue_info[queue_name]


# Create a singleton instance
queue_manager = QueueManager()
