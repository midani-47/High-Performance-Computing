from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional
from datetime import datetime, timedelta

from .models import TokenData, QueueRole
from .logger import logger
from .config import config

# OAuth2 from Fastapi for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load JWT configuration from config file
SECRET_KEY = config.get("jwt_secret_key")
ALGORITHM = config.get("jwt_algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = config.get("jwt_expiration_minutes", 30)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creating a JWT access token
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time
    
    Returns:
        JWT token as string
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Create and return the token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Get the current user from the provided JWT token
    
    Args:
        token: JWT token from authorization header
    
    Returns:
        TokenData object with user information
    
    Raising:
        HTTPException: If token is invalid or expired
    """
    # Exception
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},  # Authenticate header
    )
    
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "agent")
        
        # Get current user
        if username is None:
            logger.warning("Missing username in token")
            raise credentials_exception
        
        # Create and return token data
        token_data = TokenData(
            username=username,
            role=QueueRole(role)
        )
        return token_data
    except JWTError as e:
        logger.warning(f"JWT error: {str(e)}")
        raise credentials_exception


async def validate_admin_privileges(token_data: TokenData = Depends(get_current_user)):
    """
    Validate that the current user has admin privileges
    
    Args:
        token_data: TokenData object from get_current_user
    
    Raises:
        HTTPException: If user doesn't have admin role
    """
    if token_data.role != QueueRole.ADMIN:
        logger.warning(f"User {token_data.username} attempted admin operation without permissions")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough privileges"
        )
    return token_data



async def validate_agent_or_admin_privileges(token_data: TokenData = Depends(get_current_user)):
    """
    Validate that the current user has agent or admin privileges
    
    Args:
        token_data: TokenData object from get_current_user
    
    Raises:
        HTTPException: If user doesn't have agent or admin role
    """
    # note that this is only valid for admins or agents
    if token_data.role not in [QueueRole.AGENT, QueueRole.ADMIN]:
        logger.warning(f"User {token_data.username} attempted agent operation without permissions")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough privileges"
        )
    return token_data
