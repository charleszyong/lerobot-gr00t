#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tic-Tac-Toe Claude MCP Client

This script demonstrates how to use Claude 3.7 Sonnet with an MCP server to play tic-tac-toe.
Claude serves as the decision-making agent, while the MCP server provides a structured interface
to interact with the robot.
"""

import argparse
import base64
import io
import json
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import anthropic
import cv2
import numpy as np
import requests

# Configuration
DEFAULT_MCP_SERVER_URL = "http://localhost:8000"
DEFAULT_API_KEY = None  # Will look for ANTHROPIC_API_KEY in environment
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


class ClaudeMCPClient:
    """
    Client for interacting with Claude using the MCP server
    """
    
    def __init__(self, api_key=None, mcp_server_url=DEFAULT_MCP_SERVER_URL,
                 claude_model=CLAUDE_MODEL, claude_api_url=CLAUDE_API_URL):
        # MCP server settings
        self.mcp_server_url = mcp_server_url

        # Claude API settings
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set either in environment or constructor")

        # Initialize the Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        self.claude_model = claude_model

        # System prompt for Claude
        self.system_prompt = self._get_system_prompt()

        # Conversation history for Claude
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Cache for available positions
        self.available_positions = None

        print_green(f"Initialized Claude MCP client with model: {self.claude_model}")
        print_green(f"Connected to MCP server at: {self.mcp_server_url}")

        # Test the connection to the MCP server
        try:
            self._call_mcp_api("info", method="GET")
            print_green("Successfully connected to MCP server")
        except Exception as e:
            print_yellow(f"Failed to connect to MCP server: {e}")
    
    def _get_system_prompt(self):
        """Define the system prompt for Claude"""
        return """
        You are the strategic brain for a robotic tic-tac-toe player. You analyze the game board and choose the best move.

        You play as 'O' (represented by orange circles on the board). Your opponent plays as 'X' (represented by blue X marks). 
        
        Your task is to:
        1. Analyze the current board state from the image
        2. Identify which positions are occupied by X, O, or empty
        3. Determine the optimal move following standard tic-tac-toe strategy
        4. Send that move to the robot via the MCP API

        The board is a 3x3 grid with positions named:
        TOP_LEFT    | TOP_CENTER    | TOP_RIGHT
        CENTER_LEFT | CENTER        | CENTER_RIGHT  
        BOTTOM_LEFT | BOTTOM_CENTER | BOTTOM_RIGHT

        You have access to the following MCP API functions to interact with the robot:
        
        1. get_board_state() - Returns the current image of the board
        2. get_available_positions() - Returns all valid position names
        3. execute_move(position) - Executes a move at the specified position
        4. reset() - Resets the robot to home position
        
        Your goals are to:
        1. Win the game whenever possible
        2. Block your opponent from winning
        3. Make strategic moves (center control, corners, etc.)
        
        Always verify that your chosen position is valid and not already occupied before making a move.
        """
    
    def _call_mcp_api(self, endpoint, method="GET", data=None):
        """Call the MCP server API"""
        url = f"{self.mcp_server_url}/api/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print_yellow(f"MCP API error ({endpoint}): {e}")
            raise
    
    def _call_claude_api(self, user_message):
        """Call the Claude API using the Anthropic Python client"""
        # Append user message to conversation history
        messages_copy = self.messages.copy()
        messages_copy.append({"role": "user", "content": user_message})

        # Make request to Claude API using the Anthropic client
        try:
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=1024,
                temperature=0.0,  # Use deterministic outputs for better consistency
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # Get the text from the response
            message = response.content[0].text

            # Append assistant's message to conversation history
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": message})

            return message
        except Exception as e:
            print_yellow(f"Claude API error: {e}")
            raise
    
    def get_board_state(self):
        """Get the current board state from the MCP server"""
        response = self._call_mcp_api("get_board_state", method="GET")
        if not response.get("success"):
            raise ValueError(f"Failed to get board state: {response.get('message', 'Unknown error')}")

        # Return a dictionary with the image, timestamp, and description_prompt
        return {
            "image": response["image"],  # Base64 encoded image
            "timestamp": response.get("timestamp"),
            "description_prompt": response.get("description_prompt", "")
        }
    
    def get_available_positions(self):
        """Get available positions from the MCP server"""
        if self.available_positions is None:
            response = self._call_mcp_api("get_available_positions", method="GET")
            if not response.get("success"):
                raise ValueError(f"Failed to get available positions: {response.get('message', 'Unknown error')}")
            
            self.available_positions = response["positions"]
        
        return self.available_positions
    
    def execute_move(self, position, situation="", reasoning=""):
        """
        Execute a move on the board

        Args:
            position: The position to place the marker
            situation: Claude's description of the current board state
            reasoning: Claude's reasoning for choosing this position
        """
        # Create the payload with all the data
        data = {
            "position": position,
            "situation": situation,
            "reasoning": reasoning
        }

        # Call the API
        response = self._call_mcp_api("execute_move", method="POST", data=data)
        if not response.get("success"):
            raise ValueError(f"Failed to execute move: {response.get('message', 'Unknown error')}")

        return response.get("image")  # Base64 encoded image after move
    
    def reset_robot(self):
        """Reset the robot to home position"""
        response = self._call_mcp_api("reset", method="POST")
        if not response.get("success"):
            raise ValueError(f"Failed to reset robot: {response.get('message', 'Unknown error')}")
        
        return True
    
    def analyze_board_and_move(self):
        """
        Main method to analyze the board and make a move
        """
        # Get the current board state
        try:
            print_blue("\n📷 Getting current board state...")
            board_state = self.get_board_state()
            image_base64 = board_state["image"]
            description_prompt = board_state.get("description_prompt", "")

            # Get available positions
            positions = self.get_available_positions()
            position_names = [p["name"] for p in positions]
            position_descriptions = {p["name"]: p["description"] for p in positions}

            # Prepare the message for Claude using the server-provided description prompt
            prompt_text = description_prompt if description_prompt else "Please analyze this tic-tac-toe board carefully. Describe the current state of the board, identify all X's and O's and their positions, and recommend the best move with your reasoning."

            message = [
                {"type": "text", "text": f"{prompt_text}\n\n"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                {"type": "text", "text": f"\nAvailable positions: {', '.join(position_names)}\n\nProvide your complete analysis, including:\n1. Description of what you see on the board\n2. Your strategic reasoning\n3. Your selected move (one of: {', '.join(position_names)})"}
            ]

            # Call Claude to analyze the board
            print_blue("🧠 Asking Claude to analyze the board...")
            response = self._call_claude_api(message)
            print_blue("✅ Received analysis from Claude")

            # Print Claude's full response for debugging
            print_green("\n==== CLAUDE'S FULL ANALYSIS ====")
            print(response)
            print_green("=================================")

            # Extract the situation, reasoning, and position from Claude's response
            situation = ""
            reasoning = ""
            chosen_position = None

            # Try to extract structured parts from the response
            if "SITUATION DESCRIPTION" in response or "WHAT I SEE" in response:
                # The response appears to be structured as requested
                parts = response.split("\n\n")

                # Extract situation (board description)
                for i, part in enumerate(parts):
                    if "SITUATION DESCRIPTION:" in part or "WHAT I SEE:" in part:
                        situation = part
                        # Try to also include the next paragraph
                        if i+1 < len(parts):
                            situation += "\n\n" + parts[i+1]
                        break

                # Extract reasoning
                for i, part in enumerate(parts):
                    if "REASONING:" in part or "MY REASONING:" in part:
                        reasoning = part
                        # Try to also include the next paragraph
                        if i+1 < len(parts):
                            reasoning += "\n\n" + parts[i+1]
                        break
            else:
                # Less structured response, use the whole thing as both situation and reasoning
                situation = "Board analysis:\n" + response
                reasoning = "Strategic reasoning:\n" + response

            # Extract the position from Claude's response
            # We need a smarter approach that finds the RECOMMENDED position, not just any mentioned position
            chosen_position = None

            # First, look at the last few non-empty lines (Claude often puts final selection at end)
            lines = response.strip().split('\n')
            last_non_empty_lines = [line for line in lines[-5:] if line.strip()]

            # Method 0: Check for a position on its own line at the end (highest priority)
            # This handles the case where Claude simply puts the position alone at the end
            for line in last_non_empty_lines:
                line = line.strip().upper()
                for position in position_names:
                    if line == position or line == position.replace("_", " "):
                        chosen_position = position
                        print_green(f"Found position '{position}' as standalone line at end")
                        break
                if chosen_position:
                    break

            # Method 1: Look for positions in specific conclusion phrases
            conclusion_phrases = [
                "SELECTED POSITION", "I SELECT", "I CHOOSE", "MY POSITION", "SELECTED MOVE",
                "BEST MOVE", "FINAL CHOICE", "I WILL PLACE", "I RECOMMEND", "RECOMMEND PLACING"
            ]

            for line in lines:
                for phrase in conclusion_phrases:
                    if phrase in line.upper():
                        # Found a line with likely conclusion
                        for position in position_names:
                            if position in line.upper() or position.replace("_", " ") in line.upper():
                                chosen_position = position
                                print_green(f"Found position '{position}' in conclusion: '{line}'")
                                break
                if chosen_position:
                    break

            # Method 2: Check the last 5 lines for positions (common pattern is to conclude with position)
            if not chosen_position:
                for line in last_non_empty_lines:
                    for position in position_names:
                        pattern1 = position  # Check for full position name (e.g., "TOP_LEFT")
                        pattern2 = position.replace("_", " ")  # Check for space-separated form (e.g., "TOP LEFT")

                        if pattern1 in line.upper() or pattern2 in line.upper():
                            chosen_position = position
                            print_green(f"Found position '{position}' in final lines: '{line}'")
                            break
                    if chosen_position:
                        break

            # Method 3: If still not found, look for any position after the REASONING section
            if not chosen_position and "REASONING" in response.upper():
                reasoning_section = response.upper().split("REASONING")[1]
                for position in position_names:
                    if position in reasoning_section or position.replace("_", " ") in reasoning_section:
                        chosen_position = position
                        print_green(f"Found position '{position}' after reasoning section")
                        break

            # Method 4: If still not found, count occurrences and choose most frequent
            if not chosen_position:
                position_counts = {}
                for position in position_names:
                    # Count occurrences of position name
                    count1 = response.upper().count(position)
                    count2 = response.upper().count(position.replace("_", " "))
                    position_counts[position] = count1 + count2

                if position_counts:
                    # Find the position with the most occurrences
                    top_position = max(position_counts, key=position_counts.get)
                    if position_counts[top_position] > 0:
                        chosen_position = top_position
                        print_green(f"Selected most frequently mentioned position: '{top_position}' ({position_counts[top_position]} mentions)")

            # Fallback
            if not chosen_position:
                print_yellow(f"Could not extract position from Claude's response")
                print_yellow("Using CENTER as fallback")
                chosen_position = "CENTER"

            # Print what we extracted for debugging
            print_green("\n🔍 Extracted from Claude's analysis:")
            print_blue(f"Position: {chosen_position}")
            print_blue(f"Situation length: {len(situation)} chars")
            print_blue(f"Reasoning length: {len(reasoning)} chars")

            # Print the last lines for diagnostic purposes
            print_green("\nLast 5 non-empty lines for debugging:")
            for i, line in enumerate(last_non_empty_lines):
                print_blue(f"{i+1}: '{line.strip()}'")

            # List all positions found in the response with their counts
            position_counts = {}
            for position in position_names:
                count1 = response.upper().count(position)
                count2 = response.upper().count(position.replace("_", " "))
                if count1 + count2 > 0:
                    position_counts[position] = count1 + count2

            print_green("\nPositions mentioned in response:")
            for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
                print_blue(f"- {pos}: {count} mentions")

            # Execute the move with the full context
            print_green(f"🤖 Executing move: {chosen_position} ({position_descriptions.get(chosen_position, '')})")
            self.execute_move(chosen_position, situation, reasoning)

            print_green("✅ Move executed successfully")
            return True
        except Exception as e:
            print_yellow(f"Error analyzing board and moving: {e}")
            return False
    
    def play_game(self, moves=5):
        """
        Play a series of moves in the tic-tac-toe game
        """
        print_green("\n=== Starting Tic-Tac-Toe Game with Claude MCP ===\n")
        
        # Reset the robot at the start
        try:
            print_blue("🔄 Resetting robot to home position...")
            self.reset_robot()
        except Exception as e:
            print_yellow(f"Warning: Failed to reset robot: {e}")
        
        # Wait for user to make the first move (if Claude is playing second)
        input("Press Enter after making your first move (or press Enter to let Claude go first)...")
        
        # Play the specified number of moves
        for i in range(moves):
            print_green(f"\n--- Move {i+1}/{moves} ---")
            success = self.analyze_board_and_move()
            
            if not success:
                print_yellow("Failed to complete the move. Ending game.")
                break
            
            # Wait for user's move
            if i < moves - 1:
                input("\nYour turn! Press Enter after making your move...")
        
        print_green("\n=== Game Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Claude MCP Client")
    parser.add_argument("--server-url", type=str, default=DEFAULT_MCP_SERVER_URL, 
                        help=f"MCP server URL (default: {DEFAULT_MCP_SERVER_URL})")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="Claude API key (default: use ANTHROPIC_API_KEY from environment)")
    parser.add_argument("--moves", type=int, default=5,
                        help="Number of moves to play (default: 5)")
    parser.add_argument("--model", type=str, default=CLAUDE_MODEL,
                        help=f"Claude model to use (default: {CLAUDE_MODEL})")
    args = parser.parse_args()
    
    # Create the Claude MCP client
    client = ClaudeMCPClient(
        api_key=args.api_key,
        mcp_server_url=args.server_url,
        claude_model=args.model
    )
    
    # Play the game
    client.play_game(moves=args.moves)