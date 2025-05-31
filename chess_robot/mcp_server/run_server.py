#!/usr/bin/env python3
"""
Run the Chess Robot MCP Server
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from chess_robot.mcp_server.server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main()) 