#!/bin/bash
# Script to start the testing environment for the MPI prediction service

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MPI Prediction Service Testing Environment ===${NC}"
echo -e "${BLUE}This script will help you set up the testing environment${NC}"

# Check if the queue service from Assignment 3 is available
QUEUE_SERVICE_DIR="../assignment 3/queue_service"
if [ ! -d "$QUEUE_SERVICE_DIR" ]; then
    echo -e "${RED}Error: Queue service directory not found at $QUEUE_SERVICE_DIR${NC}"
    echo -e "${YELLOW}Please make sure the Assignment 3 queue service is available${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    echo -e "${YELLOW}Please install Docker and Docker Compose first${NC}"
    exit 1
fi

# Function to check if a port is in use
port_in_use() {
    lsof -i:$1 &> /dev/null
}

# Check if ports are already in use
if port_in_use 7500; then
    echo -e "${YELLOW}Port 7500 is already in use. Queue service might be running.${NC}"
else
    echo -e "${GREEN}Starting queue service from Assignment 3...${NC}"
    cd "../assignment 3" || exit 1
    docker-compose up -d
    cd - || exit 1
    
    # Wait for queue service to start
    echo -e "${YELLOW}Waiting for queue service to start...${NC}"
    sleep 5
fi

# Start the web UI
if port_in_use 7600; then
    echo -e "${YELLOW}Port 7600 is already in use. Web UI might be running.${NC}"
else
    echo -e "${GREEN}Starting web UI...${NC}"
    python web_ui.py &
    WEB_UI_PID=$!
    
    # Wait for web UI to start
    echo -e "${YELLOW}Waiting for web UI to start...${NC}"
    sleep 2
fi

echo -e "${GREEN}Testing environment is ready!${NC}"
echo -e "${BLUE}Queue Service: http://localhost:7500${NC}"
echo -e "${BLUE}Web UI: http://localhost:7600${NC}"

echo -e "\n${YELLOW}Instructions:${NC}"
echo -e "1. Open the Web UI in your browser: ${BLUE}http://localhost:7600${NC}"
echo -e "2. Use the Web UI to set up queues and push sample transactions"
echo -e "3. Start the MPI prediction service using one of these methods:"
echo -e "   - Local mode: ${BLUE}./start_service.sh${NC}"
echo -e "   - Docker mode: ${BLUE}docker-compose up${NC}"
echo -e "4. Use the Web UI to monitor prediction results"

echo -e "\n${YELLOW}Press Ctrl+C to stop the Web UI${NC}"
wait $WEB_UI_PID
