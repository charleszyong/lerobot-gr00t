#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example client for interacting with the Tic-Tac-Toe MCP Server

This script demonstrates how to:
1. Get the current board state from the MCP server
2. Display the image to Claude
3. Ask Claude to analyze the board using the server-provided prompt
4. Extract Claude's situation analysis and reasoning
5. Send the move command back to the MCP server

Usage:
python tictac_client_example.py

Note: This is for demonstration purposes only. In a real setup, you would
integrate this with the MCP system directly.
"""

import requests
import json
import base64
from PIL import Image
import io
import argparse

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:8000"

def get_board_state(server_url):
    """Get the current board state from the MCP server"""
    try:
        response = requests.get(f"{server_url}/api/get_board_state")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                # Extract the image, timestamp, and description prompt
                img_base64 = data["image"]
                timestamp = data["timestamp"]
                description_prompt = data["description_prompt"]
                
                # Decode the image
                img_data = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_data))
                
                return {
                    "success": True,
                    "image": img,
                    "image_base64": img_base64,
                    "timestamp": timestamp,
                    "description_prompt": description_prompt
                }
            else:
                print(f"Error from server: {data.get('message', 'Unknown error')}")
                return {"success": False, "error": data.get("message", "Unknown error")}
        else:
            print(f"Error response from server: {response.status_code}")
            return {"success": False, "error": f"HTTP error {response.status_code}"}
    except Exception as e:
        print(f"Exception getting board state: {e}")
        return {"success": False, "error": str(e)}

def execute_move(server_url, position, situation="", reasoning=""):
    """Execute a move on the board"""
    try:
        # Prepare the payload with Claude's analysis
        payload = {
            "position": position,
            "situation": situation,
            "reasoning": reasoning
        }
        
        # Send the request to the server
        response = requests.post(
            f"{server_url}/api/execute_move",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"Successfully executed move at {position}")
                # Decode the updated image if provided
                if "image" in data and data["image"]:
                    img_data = base64.b64decode(data["image"])
                    updated_img = Image.open(io.BytesIO(img_data))
                    return {
                        "success": True,
                        "message": data.get("message", ""),
                        "image": updated_img
                    }
                return {"success": True, "message": data.get("message", "")}
            else:
                print(f"Error from server: {data.get('message', 'Unknown error')}")
                return {"success": False, "error": data.get("message", "Unknown error")}
        else:
            print(f"Error response from server: {response.status_code}")
            return {"success": False, "error": f"HTTP error {response.status_code}"}
    except Exception as e:
        print(f"Exception executing move: {e}")
        return {"success": False, "error": str(e)}

def reset_robot(server_url):
    """Reset the robot to home position"""
    try:
        response = requests.post(f"{server_url}/api/reset")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print("Robot reset successfully")
                return {"success": True}
            else:
                print(f"Error from server: {data.get('message', 'Unknown error')}")
                return {"success": False, "error": data.get("message", "Unknown error")}
        else:
            print(f"Error response from server: {response.status_code}")
            return {"success": False, "error": f"HTTP error {response.status_code}"}
    except Exception as e:
        print(f"Exception resetting robot: {e}")
        return {"success": False, "error": str(e)}

def get_available_positions(server_url):
    """Get available positions from the server"""
    try:
        response = requests.get(f"{server_url}/api/get_available_positions")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                positions = data["positions"]
                return {"success": True, "positions": positions}
            else:
                print(f"Error from server: {data.get('message', 'Unknown error')}")
                return {"success": False, "error": data.get("message", "Unknown error")}
        else:
            print(f"Error response from server: {response.status_code}")
            return {"success": False, "error": f"HTTP error {response.status_code}"}
    except Exception as e:
        print(f"Exception getting positions: {e}")
        return {"success": False, "error": str(e)}

def simulate_claude_analysis(board_state):
    """
    This function simulates Claude's analysis of the board.
    
    In a real MCP implementation, you would:
    1. Send the image to Claude
    2. Include the description_prompt from the server
    3. Parse Claude's response to extract the situation, reasoning, and position
    
    Here we're just returning a simple example analysis.
    """
    # Get the description prompt from the server
    prompt = board_state.get("description_prompt", "")
    
    print("\n== INSTRUCTIONS FOR CLAUDE ==")
    print(prompt)
    print("\n== END INSTRUCTIONS ==\n")
    
    print("In a real implementation, you would:")
    print("1. Show the board image to Claude")
    print("2. Send the above prompt to Claude")
    print("3. Parse Claude's response to extract:")
    print("   - situation: Claude's description of the board state")
    print("   - reasoning: Claude's reasoning for the move")
    print("   - position: The selected position (CENTER, TOP_LEFT, etc.)")
    
    # Example Claude analysis - in real usage this would come from Claude
    situation = """
I can see a tic-tac-toe board with a 3x3 grid. 
The board currently has:
- X (human) in the TOP_LEFT position
- O (robot/circle) in the CENTER position
- The remaining positions are empty (TOP_RIGHT, CENTER_TOP, CENTER_LEFT, 
  CENTER_RIGHT, BOTTOM_LEFT, CENTER_BOTTOM, BOTTOM_RIGHT)
    """
    
    reasoning = """
Based on the current board state, the human player has placed an X in the top-left corner,
and the robot has responded by placing an O in the center. This is a standard opening.

The best strategic move now would be to place a circle (O) in the BOTTOM_RIGHT corner,
which accomplishes two things:
1. It prevents the human from creating a diagonal line from TOP_LEFT to BOTTOM_RIGHT
2. It creates two potential winning lines for the robot (CENTER to BOTTOM_RIGHT and
   a diagonal opportunity)

By playing in the BOTTOM_RIGHT corner, I force the human to defend against a potential win,
rather than being able to build their own winning strategy.
    """
    
    position = "BOTTOM_RIGHT"
    
    return {
        "situation": situation,
        "reasoning": reasoning,
        "position": position
    }

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe MCP Client Example")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER_URL,
                        help=f"MCP server URL (default: {DEFAULT_SERVER_URL})")
    args = parser.parse_args()
    
    # Get the server URL
    server_url = args.server
    
    # Step 1: Reset the robot to start fresh
    print("\n1. Resetting robot to home position...")
    reset_result = reset_robot(server_url)
    if not reset_result["success"]:
        print(f"Failed to reset robot: {reset_result.get('error', 'Unknown error')}")
        return
    
    # Step 2: Get the available positions
    print("\n2. Getting available positions...")
    positions_result = get_available_positions(server_url)
    if not positions_result["success"]:
        print(f"Failed to get positions: {positions_result.get('error', 'Unknown error')}")
        return
    
    print(f"Available positions: {json.dumps(positions_result['positions'], indent=2)}")
    
    # Step 3: Get the current board state
    print("\n3. Getting current board state...")
    board_state = get_board_state(server_url)
    if not board_state["success"]:
        print(f"Failed to get board state: {board_state.get('error', 'Unknown error')}")
        return
    
    # Step 4: Simulate Claude's analysis
    print("\n4. Simulating Claude's analysis of the board...")
    analysis = simulate_claude_analysis(board_state)
    
    # Print the analysis
    print("\nClaudet's Analysis:")
    print(f"Situation: {analysis['situation']}")
    print(f"Reasoning: {analysis['reasoning']}")
    print(f"Selected Position: {analysis['position']}")
    
    # Step 5: Execute the move
    print("\n5. Executing the move...")
    move_result = execute_move(
        server_url,
        analysis["position"],
        analysis["situation"],
        analysis["reasoning"]
    )
    
    if not move_result["success"]:
        print(f"Failed to execute move: {move_result.get('error', 'Unknown error')}")
        return
    
    print("\nMove executed successfully!")
    print(f"Server response: {move_result.get('message', '')}")
    
    print("\nDone! This example demonstrates the full client-server interaction flow.")

if __name__ == "__main__":
    main()