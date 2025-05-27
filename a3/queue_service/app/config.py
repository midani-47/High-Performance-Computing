import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    "max_messages_per_queue": 1000,    # Maximum number of messages per queue
    "persist_interval_seconds": 60,    # Time until storing queue state to disk
    "storage_path": "./queue_data",
    "port": 7500,
    "host": "localhost",
    "log_level": "INFO",
    "jwt_secret_key": "your-secret-key-change-in-production",  # Secret for JWT auth tokens
    "jwt_algorithm": "HS256",
    "jwt_expiration_minutes": 30     # Auth Token valid time
}


class Config:
    """
    Configuration handler for the Queue Service
    
    Reads configuration from JSON file or uses default values.
    Configuration file path can be set via QUEUE_CONFIG_PATH environment variable
    """
    
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self._load_config()

    
    def _load_config(self):
        """Load configuration from file if it exists"""
        config_path = os.environ.get("QUEUE_CONFIG_PATH", "config.json")
        path = Path(config_path)
        
        if path.exists():
            try:
                with open(path, "r") as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                print(f"Configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading configuration: {str(e)}")
                print("Using default configuration")
                
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key"""
        return self._config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.copy()
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._config[key] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Saving current configuration to file"""
        if path is None:
            path = os.environ.get("QUEUE_CONFIG_PATH", "config.json")
        
        try:
            with open(path, "w") as f:
                json.dump(self._config, f, indent=4)
            print(f"Configuration saved to {path}")
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")



config = Config()  # creating gloval instance for application
