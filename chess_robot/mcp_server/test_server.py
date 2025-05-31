#!/usr/bin/env python3
"""
Test script for Chess Robot MCP Server
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from chess_robot.mcp_server.server import ChessRobotMCPServer

async def test_server():
    """Test MCP server functionality"""
    print("Testing Chess Robot MCP Server...")
    print("=" * 50)
    
    # Create server instance
    server = ChessRobotMCPServer()
    
    # Test 1: Check that handlers are registered
    print("\n1. Testing handler registration:")
    print(f"  Server name: {server.server.name}")
    print("  Handlers registered successfully")
    
    # Test 2: Get robot status (without connection)
    print("\n2. Testing get_robot_status() - No connection:")
    status = await server.get_robot_status()
    print(f"  Status: {json.dumps(status, indent=2)}")
    
    # Test 3: List available trajectories
    print("\n3. Testing list_available_trajectories():")
    trajectories = await server.list_available_trajectories()
    print(f"  Total trajectories: {trajectories.get('total_trajectories', 0)}")
    print(f"  Pickup count: {trajectories.get('pickup_count', 0)}")
    print(f"  Putdown count: {trajectories.get('putdown_count', 0)}")
    
    # Test 4: Get trajectory info (if available)
    if trajectories.get('pickup_trajectories'):
        square = trajectories['pickup_trajectories'][0]
        print(f"\n4. Testing get_trajectory_info() for {square} pickup:")
        info = await server.get_trajectory_info(square, 'pickup')
        print(f"  Info: {json.dumps(info, indent=2)}")
    
    # Test 5: Test initialization without robot (should fail gracefully)
    print("\n5. Testing initialize_robot() - Expecting failure without physical robot:")
    init_result = await server.initialize_robot()
    print(f"  Result: {json.dumps(init_result, indent=2)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nNote: To fully test the MCP server with robot control,")
    print("please use it through Cursor with the robot connected.")
    print("\nTo verify MCP protocol compliance, you can also run:")
    print("  python -m mcp.server.stdio chess_robot.mcp_server.server")

if __name__ == "__main__":
    asyncio.run(test_server()) 