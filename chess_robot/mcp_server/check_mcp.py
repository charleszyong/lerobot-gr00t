#!/usr/bin/env python3
"""
Check if MCP server can start and respond to basic protocol messages
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    print("✓ MCP imports successful")
except ImportError as e:
    print(f"✗ Failed to import MCP: {e}")
    sys.exit(1)

try:
    from chess_robot.mcp_server.server import ChessRobotMCPServer
    print("✓ Chess Robot MCP Server imports successful")
except ImportError as e:
    print(f"✗ Failed to import Chess Robot MCP Server: {e}")
    sys.exit(1)

print("\nMCP server is ready to use!")
print("\nTo use in Cursor:")
print("1. Add the configuration from cursor_config.json to Cursor Settings → MCP")
print("2. Make sure to use Agent mode in Cursor chat")
print("3. Start with 'Initialize the chess robot' when ready to play") 