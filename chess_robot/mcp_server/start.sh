#!/bin/bash
# Start script for chess robot MCP server

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path
export PYTHONPATH="${DIR}/../..:${PYTHONPATH}"

# Activate conda environment and run server
source /home/charles/miniconda3/bin/activate so100
cd "${DIR}/../.."
exec python -m chess_robot.mcp_server.run_server 