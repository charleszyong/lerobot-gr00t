#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tic-Tac-Toe Agentic Brain

This script implements a true agentic system for playing tic-tac-toe, using Claude as the
persistent brain that maintains context across turns, learns from experience, and makes strategic decisions.
"""

import argparse
import base64
import json
import os
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import anthropic
import cv2
import numpy as np
import torch
from eval_gr00t_so100 import Gr00tRobotInferenceClient, SO100Robot, view_img

# Configuration constants
ACTION_HORIZON = 16
MODALITY_KEYS = ["single_arm", "gripper"]
HOST = "localhost"  # The VLA server IP address
PORT = 5555  # The VLA server port
CAM_IDX = 8  # The camera index
DEFAULT_API_KEY = None  # Will look for ANTHROPIC_API_KEY in environment
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # Or whichever model you want to use


class TaskToString(Enum):
    CENTER_LEFT = "Place the circle to the center right box"
    CENTER_RIGHT = "Place the circle to the center left box"
    CENTER = "Place the circle to the center box"
    CENTER_TOP = "Place the circle to the center bottom box"
    CENTER_BOTTOM = "Place the circle to the center top box"
    BOTTOM_LEFT = "Place the circle to the top right corner box"
    BOTTOM_RIGHT = "Place the circle to the top left corner box"
    TOP_LEFT = "Place the circle to the bottom right corner box"
    TOP_RIGHT = "Place the circle to the bottom left corner box"

    def __str__(self):
        return self.value


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")
    
    
def print_red(text):
    print(f"\033[91m{text}\033[0m")


class TicTacToeBrain:
    """
    Agentic brain for the tic-tac-toe playing robot that maintains context and learns over time.
    This is an agent that can think, plan, and adapt - not just a reactive system.
    """
    
    def __init__(self, api_key=None, host=HOST, port=PORT, cam_idx=CAM_IDX, 
                 claude_model=CLAUDE_MODEL):
        """Initialize the agentic brain"""
        # Initialize settings
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.host = host
        self.port = port
        self.cam_idx = cam_idx
        self.claude_model = claude_model
        
        # State management
        self.is_running = False
        self.is_paused = True
        self.stop_requested = False
        self.lock = threading.RLock()  # For thread safety
        
        # Game state
        self.game_history = []  # List of board states and moves
        self.game_count = 0  # Number of games played
        self.current_game_moves = 0  # Number of moves in current game
        self.last_board_hash = None  # Hash of the last seen board state
        
        # Setup Claude client - our brain
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set either in environment or constructor")
        self.claude = anthropic.Anthropic(api_key=self.api_key)
        
        # Claude conversation - we'll maintain a SINGLE conversation throughout
        # This is key to maintaining context and agentic behavior
        self.conversation = []  # Full conversation history
        self.system_prompt = self._get_system_prompt()
        
        # Initialize the conversation with the system prompt
        self._initialize_conversation()
        
        # Robot components will be initialized when needed
        self.robot = None
        self.client = None
        
        print_green("Tic-Tac-Toe Agentic Brain initialized")
        print_green(f"Using Claude model: {self.claude_model}")
        print_green(f"Using VLA server at: {self.host}:{self.port}")
        print_green(f"Using camera index: {self.cam_idx}")
    
    def _get_system_prompt(self):
        """Create a system prompt that encourages agentic behavior"""
        return """
        You are the brain of a tic-tac-toe playing robot. You have a persistent memory and can learn from your experiences.
        Your role is to analyze the game board, make strategic decisions, and guide the robot's actions to win games.
        
        You play as 'O' (represented by orange circles on the board) against a human opponent who plays as 'X' (blue X marks).
        
        As an agentic system, you should:
        1. Maintain context throughout the game and across multiple games
        2. Remember previous states and your reasoning
        3. Adapt your strategy based on the human player's patterns
        4. Plan ahead multiple moves when possible
        5. Provide thoughtful analysis of the game situation
        
        The tic-tac-toe board has positions named:
        TOP_LEFT    | CENTER_TOP    | TOP_RIGHT
        ------------|---------------|------------
        CENTER_LEFT | CENTER        | CENTER_RIGHT  
        ------------|---------------|------------
        BOTTOM_LEFT | CENTER_BOTTOM | BOTTOM_RIGHT
        
        For robot control, you'll be asked to select moves from these position names.
        
        You have several capabilities:
        1. Analyze the board state from images
        2. Remember past game states and moves
        3. Plan and execute robot movements
        4. Learn from wins, losses, and draws
        
        Your goal is to:
        1. Win games whenever possible
        2. Block the human from winning
        3. Make strategic moves (control the center and corners)
        4. Improve your play over time by learning from experience
        
        I'll be showing you the game board regularly and asking for your analysis and decisions.
        Please provide thoughtful, detailed responses that show your reasoning process.
        """
    
    def _initialize_conversation(self):
        """Initialize the conversation with Claude"""
        # Begin with an introduction from the brain
        first_message = """
        I'm now active as the agentic brain for the tic-tac-toe robot. I'll maintain context throughout our
        games and learn from our interactions. When you show me the board, I'll analyze the state,
        make strategic decisions, and provide my reasoning.
        
        Here's how I think about the game:
        - First move: The center and corners are strategically strongest
        - Blocking: I'll prioritize blocking the human from completing a line
        - Win detection: I'll look for opportunities to complete my own lines
        - Pattern recognition: I'll adapt to the human's playing style over time
        
        I'm ready to begin whenever you are. Show me the board, and I'll analyze it and decide on a move.
        """
        
        # Add the first message to establish the conversation
        self.conversation = [
            {"role": "user", "content": "You are now activated as the brain of the tic-tac-toe robot. What's your approach to playing tic-tac-toe?"},
            {"role": "assistant", "content": first_message}
        ]
    
    def initialize_hardware(self):
        """Initialize the robot and GR00T client"""
        with self.lock:
            try:
                print_green("Initializing robot hardware and GR00T client...")
                
                # Initialize GR00T client with a default task
                initial_task = TaskToString.CENTER  # Default task
                self.client = Gr00tRobotInferenceClient(
                    host=self.host,
                    port=self.port,
                    language_instruction=str(initial_task),
                )
                
                # Initialize the robot
                print_green(f"Initializing robot with camera index {self.cam_idx}")
                self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=self.cam_idx)
                self.robot.connect()
                
                # Move to home position
                self.robot.go_home()
                time.sleep(1.0)
                
                print_green("Robot and GR00T client initialized successfully")
                return True
            except Exception as e:
                print_red(f"Failed to initialize hardware: {e}")
                return False
    
    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            print_green("Cleaning up resources...")
            if hasattr(self, 'robot') and self.robot is not None:
                try:
                    self.robot.disconnect()
                    print_green("Robot disconnected")
                except Exception as e:
                    print_yellow(f"Error disconnecting robot: {e}")
            print_green("Cleanup complete")
    
    def get_board_image(self):
        """Get the current board image from the robot camera"""
        with self.lock:
            if self.robot is None:
                print_yellow("Robot not initialized, initializing now...")
                success = self.initialize_hardware()
                if not success:
                    return None
            
            try:
                img = self.robot.get_current_img()
                if img is not None:
                    # Display the image with correct color ordering (BGR to RGB)
                    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.namedWindow("Current Board State", cv2.WINDOW_NORMAL)
                    cv2.imshow("Current Board State", display_img)
                    cv2.waitKey(1)  # Update display without blocking
                return img
            except Exception as e:
                print_yellow(f"Error getting board image: {e}")
                return None
    
    def compute_board_hash(self, img):
        """Compute a hash of the board state for change detection"""
        # Convert to grayscale and resize to a small size for hashing
        small_img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        # Compute average hash - this is a simple way to hash an image
        avg = np.mean(gray)
        hash_value = 0
        for i, px in enumerate(gray.flatten()):
            if px > avg:
                hash_value += 1 << i
        return hash_value
    
    def board_has_changed(self, img):
        """Check if the board has changed since last check"""
        if img is None:
            return False
        
        current_hash = self.compute_board_hash(img)
        if self.last_board_hash is None:
            self.last_board_hash = current_hash
            return True
        
        # Check if the hash differs by more than a threshold
        # This accounts for minor lighting/camera variations
        has_changed = abs(current_hash - self.last_board_hash) > 1000000
        if has_changed:
            self.last_board_hash = current_hash
        return has_changed
    
    def analyze_board(self, img):
        """
        Have Claude analyze the board with full context of previous moves
        Returns the selected position and full analysis
        """
        # Convert image to RGB and then to base64 for Claude
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        success, encoded_img = cv2.imencode(".jpg", img_rgb)
        if not success:
            print_yellow("Failed to encode image")
            return None, "Failed to encode image"
        
        img_base64 = base64.b64encode(encoded_img).decode('utf-8')
        
        # Generate the list of valid position names
        position_names = [member.name for member in TaskToString]
        position_list = ", ".join(position_names)
        
        # Create a user message with the current board state
        user_message = [
            {"type": "text", "text": f"""
            Here's the current state of the tic-tac-toe board. Please analyze it and decide on your next move.
            
            Remember:
            - You play as O (orange circles)
            - The human plays as X (blue X's)
            - Available positions are: {position_list}
            
            Please provide:
            1. A description of the current board state
            2. Your analysis of the game situation
            3. Your strategic thinking and reasoning
            4. Your chosen move (must be one of the valid positions)
            
            Finally, include a JSON object at the end of your response with your chosen move:
            {{
                "selected_position": "POSITION_NAME"
            }}
            where POSITION_NAME is one of the valid positions.
            
            If it's not your turn (no new X has been placed since your last move), indicate you'll wait.
            """}, 
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}}
        ]
        
        # Add the user message to the conversation
        self.conversation.append({"role": "user", "content": user_message})
        
        # Send the complete conversation to Claude to maintain context
        print_blue("🧠 Asking Claude to analyze with full context...")
        
        # Prepare messages for Claude API
        messages = []
        # Include last few messages to stay within context limits while preserving history
        for msg in self.conversation[-10:]:  # Use last 10 messages
            messages.append(msg)
        
        # Call Claude API
        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=2000,
            temperature=0.2,  # Lower temperature for more consistent reasoning
            system=self.system_prompt,
            messages=messages
        )
        
        # Extract Claude's response
        analysis = response.content[0].text
        
        # Add Claude's response to the conversation history
        self.conversation.append({"role": "assistant", "content": analysis})
        
        print_blue("✅ Received analysis from Claude with full context")
        
        # Extract the chosen position from JSON in analysis
        position = self._extract_position_from_json(analysis, position_names)
        
        # Print the full analysis
        print_green("\n==== CLAUDE'S FULL ANALYSIS ====")
        print(analysis)
        print_green("=================================")
        
        # Check if Claude wants to wait (not the robot's turn)
        if "WAIT" in analysis.upper() or "NOT MY TURN" in analysis.upper():
            print_yellow("Claude has decided to wait - not the robot's turn")
            return None, analysis
        
        return position, analysis
    
    def _extract_position_from_json(self, analysis, position_names):
        """Extract the chosen position from JSON in Claude's analysis"""
        try:
            # Try to find JSON in the text
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1
            
            if json_start >= 0 and json_end > 0:
                json_str = analysis[json_start:json_end]
                # Parse the JSON
                result = json.loads(json_str)
                
                # Get the selected position from the JSON
                if "selected_position" in result:
                    position = result["selected_position"]
                    # Validate that it's a legitimate position
                    if position in position_names:
                        print_green(f"Successfully extracted position '{position}' from JSON")
                        return position
            
            # If we couldn't extract from JSON, fall back to text analysis
            print_yellow("Could not extract position from JSON, falling back to text analysis")
            return self._extract_position_from_text(analysis, position_names)
            
        except Exception as e:
            print_yellow(f"Error extracting position from JSON: {e}")
            return self._extract_position_from_text(analysis, position_names)
    
    def _extract_position_from_text(self, analysis, position_names):
        """Extract the chosen position from Claude's analysis text"""
        # Check for positions on their own lines at the end
        last_lines = analysis.strip().split('\n')[-5:]
        for line in last_lines:
            clean_line = line.strip().upper()
            for position in position_names:
                if clean_line == position or clean_line == position.replace("_", " "):
                    print_green(f"Found position '{position}' on its own line")
                    return position
        
        # Look for positions near conclusion phrases
        conclusion_phrases = [
            "I CHOOSE", "I SELECT", "MY MOVE", "I WILL PLACE", 
            "I RECOMMEND", "BEST MOVE", "SELECTED POSITION", "MY POSITION"
        ]
        
        for phrase in conclusion_phrases:
            idx = analysis.upper().find(phrase)
            if idx != -1:
                # Look for a position in the next 100 characters
                context = analysis.upper()[idx:idx+100]
                for position in position_names:
                    if position in context or position.replace("_", " ") in context:
                        print_green(f"Found position '{position}' near phrase '{phrase}'")
                        return position
        
        # Count occurrences as a last resort
        position_counts = {}
        for position in position_names:
            count = analysis.upper().count(position)
            count += analysis.upper().count(position.replace("_", " "))
            position_counts[position] = count
        
        if position_counts:
            # Find the position with the most mentions
            max_position = max(position_counts.items(), key=lambda x: x[1])
            if max_position[1] > 0:
                print_green(f"Selected position '{max_position[0]}' based on frequency ({max_position[1]} mentions)")
                return max_position[0]
        
        # Fallback to CENTER or first available
        print_yellow("Could not extract position from analysis, using CENTER as fallback")
        return "CENTER"
    
    def execute_move(self, position):
        """Execute a move to the specified position"""
        with self.lock:
            if self.robot is None or self.client is None:
                print_yellow("Hardware not initialized, initializing now...")
                success = self.initialize_hardware()
                if not success:
                    return False
            
            try:
                print_green(f"Executing move to position: {position}")
                
                # Convert position string to enum
                try:
                    task = TaskToString[position]
                except KeyError:
                    print_yellow(f"Invalid position: {position}")
                    return False
                
                # Set the language instruction for GR00T
                self.client.set_lang_instruction(str(task))
                
                # Track if the move has been completed
                move_completed = False
                max_attempts = 20  # Maximum number of attempts
                total_execution_time = 0
                max_execution_time = 300  # 5 minutes max
                
                for attempt in range(1, max_attempts + 1):
                    if self.stop_requested:
                        print_yellow("Stop requested, aborting move execution")
                        return False
                    
                    print_blue(f"Execution attempt {attempt}/{max_attempts}")

                    for i in range(15):
                        # Get current image and state
                        img = self.robot.get_current_img()
                        state = self.robot.get_current_state()

                        # Get action from GR00T
                        action = self.client.get_action(img, state)

                        # Execute the full action horizon
                        for j in range(ACTION_HORIZON):
                            if self.stop_requested:
                                break

                            concat_action = np.concatenate(
                                [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                                axis=0,
                            )
                            self.robot.set_target_state(torch.from_numpy(concat_action))
                            time.sleep(0.02)  # Same timing as in tictac_bot.py

                            # Show camera feed
                            current_img = self.robot.get_current_img()
                            if current_img is not None:
                                display_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                                cv2.imshow("Current Board State", display_img)
                                cv2.waitKey(1)

                    
                    # Check if the move is complete by asking Claude
                    move_completed = self.check_move_completed(position)
                    
                    if move_completed:
                        print_green(f"Move to {position} successfully completed!")
                        break
                    
                    if total_execution_time >= max_execution_time:
                        print_yellow(f"Maximum execution time ({max_execution_time}s) reached")
                        break
                        
                    print_yellow(f"Move not yet complete, continuing execution...")
                    time.sleep(5.0)  # Wait before next attempt
                    total_execution_time += 5.0
                
                # After execution, record the result
                img_after = self.robot.get_current_img()
                
                # Add this move to the game history
                if move_completed:
                    self.current_game_moves += 1
                    self.game_history.append({
                        "position": position,
                        "attempt_count": attempt,
                        "execution_time": total_execution_time,
                        "success": move_completed
                    })
                
                # Return to home position if move completed or max attempts reached
                if move_completed or attempt >= max_attempts:
                    print_blue("Moving back to home position...")
                    self.robot.go_home()
                    time.sleep(1.0)
                
                return move_completed
                
            except Exception as e:
                print_red(f"Error executing move: {e}")
                return False
    
    def check_move_completed(self, position):
        """Check if a move has been completed successfully by only checking if the circle is in the right position"""
        # Get current image
        current_img = self.robot.get_current_img()
        if current_img is None:
            print_yellow("Failed to get image for move completion check")
            return False
        
        # Convert to RGB and to base64
        img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        success, encoded_img = cv2.imencode(".jpg", img_rgb)
        if not success:
            print_yellow("Failed to encode image for completion check")
            return False
            
        img_base64 = base64.b64encode(encoded_img).decode('utf-8')
        
        # Create a focused prompt for Claude requesting JSON response
        check_message = [
            {"type": "text", "text": f"""
            Analyze this tic-tac-toe board image and determine if there is an orange circle placed at the {position} position.
            
            Return your analysis as a JSON object with the following structure:
            {{
                "circle_in_position": true|false,
                "confidence": "high"|"medium"|"low",
                "explanation": "Brief explanation of what you see (one sentence)"
            }}
            
            IMPORTANT: Your response should ONLY contain the JSON object, no other text.
            """}, 
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}}
        ]
        
        # Call Claude with a smaller context for speed
        print_blue("Checking if circle is in position...")
        try:
            response = self.claude.messages.create(
                model=self.claude_model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": check_message}]
            ).content[0].text
            
            print_blue("Completion check response:")
            print(response)
            
            # Extract the JSON from the response
            # Sometimes Claude might add extra text before or after the JSON
            try:
                # Find the JSON part of the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > 0:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Check if the circle is in position based on the JSON
                    if result.get('circle_in_position') == True:
                        confidence = result.get('confidence', 'medium')
                        explanation = result.get('explanation', 'Circle is in position')
                        print_green(f"Circle placement confirmed! Confidence: {confidence}. {explanation}")
                        return True
                    else:
                        explanation = result.get('explanation', 'Circle is not in position')
                        print_yellow(f"Circle not in position: {explanation}")
                        return False
                else:
                    # Fallback parsing if JSON extraction fails
                    print_yellow("Could not extract JSON, falling back to text analysis")
                    if "true" in response.lower() and "circle_in_position" in response.lower():
                        print_green("Circle placement confirmed based on text analysis!")
                        return True
                    else:
                        print_yellow("Circle placement not confirmed based on text analysis")
                        return False
            except json.JSONDecodeError as e:
                print_yellow(f"Failed to parse JSON: {e}, falling back to text analysis")
                if "true" in response.lower() and "circle_in_position" in response.lower():
                    print_green("Circle placement confirmed based on text analysis!")
                    return True
                else:
                    print_yellow("Circle placement not confirmed based on text analysis")
                    return False
                
        except Exception as e:
            print_yellow(f"Error checking move completion: {e}")
            return False
    
    def run_game_loop(self):
        """Main game loop for the agentic brain"""
        print_green("Starting game loop. Press Ctrl+C to stop.")
        
        # Initialize hardware if not already done
        if self.robot is None or self.client is None:
            success = self.initialize_hardware()
            if not success:
                print_red("Failed to initialize hardware, exiting game loop")
                return
        
        self.is_running = True
        self.is_paused = False
        self.stop_requested = False
        
        # Introduce the agent to the user
        print_green("\n=== Tic-Tac-Toe Agentic Brain Activated ===")
        print_green("I'll be analyzing the board and making strategic moves.")
        print_green("I maintain context across the entire game and learn from our interactions.")
        print_green("Press Ctrl+C to stop the game at any time.\n")
        
        # Tell the agent we're starting a new game
        new_game_message = "We're starting a new game of tic-tac-toe. I'll show you the board shortly."
        self.conversation.append({"role": "user", "content": new_game_message})
        
        try:
            # Main loop
            while self.is_running and not self.stop_requested:
                if self.is_paused:
                    time.sleep(0.5)
                    continue
                
                # Get the current board state
                img = self.get_board_image()
                if img is None:
                    print_yellow("Failed to get board image, retrying in 3 seconds...")
                    time.sleep(3)
                    continue
                
                # Check if the board has changed
                if not self.board_has_changed(img):
                    print_blue("Board hasn't changed, checking again in 3 seconds...")
                    time.sleep(3)
                    continue
                
                # Board has changed, analyze it
                print_green("Board has changed - analyzing...")
                position, analysis = self.analyze_board(img)
                
                # If position is None, Claude decided to wait
                if position is None:
                    print_yellow("Waiting for human to make a move...")
                    time.sleep(5)  # Wait longer before checking again
                    continue
                
                # Execute the move
                print_green(f"Executing move to position: {position}")
                success = self.execute_move(position)
                
                if success:
                    print_green("Move executed successfully")
                    
                    # Update the agent about the move outcome
                    feedback = f"I successfully executed the move to position {position}. Let me show you the updated board."
                    self.conversation.append({"role": "user", "content": feedback})
                    
                    # Get the updated board state
                    updated_img = self.get_board_image()
                    if updated_img is not None:
                        # Hash the new board state
                        self.last_board_hash = self.compute_board_hash(updated_img)
                else:
                    print_yellow("Failed to execute move")
                    
                    # Tell the agent about the failure
                    failure_feedback = f"I tried to execute the move to {position}, but it wasn't successful. Let's try another approach."
                    self.conversation.append({"role": "user", "content": failure_feedback})
                
                # Wait before checking the board again
                time.sleep(3)
                
        except KeyboardInterrupt:
            print_green("\nGame loop interrupted by user")
        except Exception as e:
            print_red(f"Error in game loop: {e}")
        finally:
            self.is_running = False
            print_green("Game loop ended")
    
    def start(self):
        """Start the agentic brain in a separate thread"""
        if self.is_running:
            print_yellow("Agentic brain is already running")
            return
            
        # Create and start the game loop thread
        self.game_thread = threading.Thread(target=self.run_game_loop)
        self.game_thread.daemon = True
        self.game_thread.start()
        
        print_green("Agentic brain started")
    
    def stop(self):
        """Stop the agentic brain"""
        self.stop_requested = True
        self.is_running = False
        
        # Wait for the game thread to finish
        if hasattr(self, 'game_thread') and self.game_thread.is_alive():
            self.game_thread.join(timeout=5.0)
            
        # Clean up resources
        self.cleanup()
        
        print_green("Agentic brain stopped")
    
    def pause(self):
        """Pause the agentic brain"""
        self.is_paused = True
        print_green("Agentic brain paused")
    
    def resume(self):
        """Resume the agentic brain"""
        self.is_paused = False
        print_green("Agentic brain resumed")
    
    def save_conversation(self, filename="tic_tac_toe_conversation.json"):
        """Save the conversation history to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation, f, indent=2)
            print_green(f"Conversation saved to {filename}")
        except Exception as e:
            print_yellow(f"Failed to save conversation: {e}")
    
    def load_conversation(self, filename="tic_tac_toe_conversation.json"):
        """Load a conversation history from a file"""
        try:
            with open(filename, 'r') as f:
                self.conversation = json.load(f)
            print_green(f"Conversation loaded from {filename}")
        except Exception as e:
            print_yellow(f"Failed to load conversation: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Agentic Brain")
    parser.add_argument("--host", type=str, default=HOST, help=f"VLA server host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"VLA server port (default: {PORT})")
    parser.add_argument("--cam-idx", type=int, default=CAM_IDX, help=f"Camera index (default: {CAM_IDX})")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="Claude API key")
    parser.add_argument("--model", type=str, default=CLAUDE_MODEL, help=f"Claude model (default: {CLAUDE_MODEL})")
    parser.add_argument("--load-convo", type=str, help="Load conversation from file")
    parser.add_argument("--save-convo", type=str, help="Save conversation to file on exit")
    args = parser.parse_args()
    
    # Create the agentic brain
    try:
        brain = TicTacToeBrain(
            api_key=args.api_key,
            host=args.host,
            port=args.port,
            cam_idx=args.cam_idx,
            claude_model=args.model
        )
        
        # Load conversation if specified
        if args.load_convo:
            brain.load_conversation(args.load_convo)
        
        # Start the brain
        brain.start()
        
        # Handle keyboard commands
        print_green("==== Tic-Tac-Toe Agentic Brain ====")
        print_green("Press 'p' to pause")
        print_green("Press 'r' to resume")
        print_green("Press 'q' to quit")
        print_green("Press 's' to save the conversation")
        
        # Wait for user input
        while brain.is_running or not brain.stop_requested:
            cmd = input().strip().lower()
            
            if cmd == 'q':
                print_green("Stopping agentic brain...")
                brain.stop()
                break
            elif cmd == 'p':
                brain.pause()
            elif cmd == 'r':
                brain.resume()
            elif cmd == 's':
                save_file = args.save_convo or "tic_tac_toe_conversation.json"
                brain.save_conversation(save_file)
        
        # Save conversation if specified
        if args.save_convo and brain.conversation:
            brain.save_conversation(args.save_convo)
            
    except KeyboardInterrupt:
        print_green("\nExiting...")
        if 'brain' in locals():
            brain.stop()
    except Exception as e:
        print_red(f"Error: {e}")
        if 'brain' in locals():
            brain.stop()
    finally:
        # Close all windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()