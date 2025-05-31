#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tic-Tac-Toe Continuous Agent

This script implements an autonomous agent that continuously monitors a tic-tac-toe board
and makes moves using Claude as the decision maker and GR00T for robotic actions.
"""

import argparse
import base64
import io
import os
import queue
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
ACTIONS_TO_EXECUTE = 10
MODALITY_KEYS = ["single_arm", "gripper"]
HOST = "localhost"  # The VLA server IP address
PORT = 5555  # The VLA server port
CAM_IDX = 8  # The camera index
DEFAULT_API_KEY = None  # Will look for ANTHROPIC_API_KEY in environment
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # Or whichever model you want to use
MONITORING_INTERVAL = 3  # Seconds between board state checks


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


class TicTacToeAgent:
    """
    Autonomous agent that continuously monitors the tic-tac-toe board and makes moves
    """
    
    def __init__(self, api_key=None, host=HOST, port=PORT, cam_idx=CAM_IDX, 
                 claude_model=CLAUDE_MODEL, monitoring_interval=MONITORING_INTERVAL):
        # Initialize settings
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.host = host
        self.port = port
        self.cam_idx = cam_idx
        self.claude_model = claude_model
        self.monitoring_interval = monitoring_interval
        
        # State management
        self.is_running = False
        self.paused = True
        self.lock = threading.RLock()
        self.last_board_state = None
        self.last_move_time = 0
        self.moves_made = 0
        
        # Claude API client
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set either in environment or constructor")
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        
        # System prompt for Claude
        self.system_prompt = self._get_system_prompt()
        
        # Command queue for thread-safe communication
        self.command_queue = queue.Queue()
        
        print_green(f"Initializing TicTacToe Agent")
        print_green(f"Using Claude model: {self.claude_model}")
        print_green(f"Using VLA server at: {self.host}:{self.port}")
        print_green(f"Using camera index: {self.cam_idx}")
        print_green(f"Monitoring interval: {self.monitoring_interval} seconds")
        
    def _get_system_prompt(self):
        """Define the system prompt for Claude"""
        return """
        You are the strategic brain for a robotic tic-tac-toe player. You analyze the game board and choose the best move.

        You play as 'O' (represented by orange circles on the board). Your opponent plays as 'X' (represented by blue X marks). 
        
        Your task is to:
        1. Analyze the current board state from the image
        2. Identify which positions are occupied by X, O, or empty
        3. Determine if the board state has changed since your last analysis
        4. Determine the optimal move following standard tic-tac-toe strategy
        5. Send that move to the robot via the API
        
        Only make a move if it's your turn (if there's a new X on the board since your last analysis).

        The board is a 3x3 grid with positions named:
        TOP_LEFT    | TOP_CENTER    | TOP_RIGHT
        CENTER_LEFT | CENTER        | CENTER_RIGHT  
        BOTTOM_LEFT | BOTTOM_CENTER | BOTTOM_RIGHT

        Your goals are to:
        1. Win the game whenever possible
        2. Block your opponent from winning
        3. Make strategic moves (center control, corners, etc.)
        
        Always verify that your chosen position is valid and not already occupied before making a move.
        """
        
    def initialize_robot(self):
        """Initialize the robot and GR00T client"""
        with self.lock:
            print_green("Initializing robot and GR00T client...")
            
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
            
    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            print_green("Cleaning up resources...")
            if hasattr(self, 'robot'):
                self.robot.disconnect()
            print_green("Cleanup complete")
            
    def get_board_state(self):
        """Get the current board state as an image"""
        with self.lock:
            try:
                img = self.robot.get_current_img()
                if img is not None:
                    # Display the image with correct color ordering (convert BGR to RGB)
                    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.namedWindow("Current Board State", cv2.WINDOW_NORMAL)
                    cv2.imshow("Current Board State", display_img)
                    cv2.waitKey(1)  # Update display without blocking
                return img
            except Exception as e:
                print_yellow(f"Error getting board state: {e}")
                return None
                
    def analyze_board(self, img):
        """
        Analyze the board using Claude and determine if a move should be made
        Returns: (should_move, position, situation, reasoning)
        """
        try:
            # Convert image from BGR to RGB for proper display in Claude
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert image to base64 for Claude
            success, encoded_img = cv2.imencode(".jpg", img_rgb)
            if not success:
                raise ValueError("Could not encode image")
            img_base64 = base64.b64encode(encoded_img).decode('utf-8')
            
            # Generate position names list
            position_names = [member.name for member in TaskToString]
            
            # Prepare the message for Claude
            description_prompt = """
            Please analyze this tic-tac-toe board image carefully and provide:

            1. SITUATION DESCRIPTION:
               - Describe the current board state in detail
               - Which positions are occupied by Xs (human)
               - Which positions are occupied by Os (robot/circles)
               - Which positions are empty
               - Label positions using the reference provided

            2. GAME ANALYSIS:
               - Who played last?
               - Whose turn is next?
               - Is there a winner? If so, who?
               - Is the game a draw?
               - Has the board state changed since my last analysis?

            3. REASONING:
               - Based on this analysis, what's the best next move?
               - Explain your reasoning strategically
               - If it's not your turn yet, indicate that you'll wait
               
            After your analysis, if it's your turn to move, provide your selected position exactly.
            If it's not your turn, state "WAIT" instead.
            """
            
            message = [
                {"type": "text", "text": f"{description_prompt}\n\n"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                {"type": "text", "text": f"\nAvailable positions: {', '.join(position_names)}\n\nProvide your complete analysis, including your final decision at the end (either a position name or WAIT)."}
            ]
            
            # Call Claude API
            print_blue("🧠 Asking Claude to analyze the board...")
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=1024,
                temperature=0.0,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": message}
                ]
            ).content[0].text
            
            print_blue("✅ Received analysis from Claude")
            
            # For debugging, print the complete analysis
            print_green("\n==== CLAUDE'S FULL ANALYSIS ====")
            print(response)
            print_green("=================================")
            
            # Extract situation, reasoning, and decision (position or WAIT)
            situation = ""
            reasoning = ""
            decision = ""
            
            # Extract structured parts from the response
            parts = response.split("\n\n")
            
            # Extract situation
            for i, part in enumerate(parts):
                if "SITUATION DESCRIPTION:" in part or "WHAT I SEE:" in part:
                    situation = part
                    if i+1 < len(parts):
                        situation += "\n\n" + parts[i+1]
                    break
            
            # Extract reasoning
            for i, part in enumerate(parts):
                if "REASONING:" in part or "MY REASONING:" in part:
                    reasoning = part
                    if i+1 < len(parts):
                        reasoning += "\n\n" + parts[i+1]
                    break
            
            # Look for "WAIT" in the response
            if "WAIT" in response.upper():
                print_green("Claude decided to WAIT - not the robot's turn yet")
                return False, None, situation, reasoning
                
            # Extract the position from Claude's response
            chosen_position = None
            last_non_empty_lines = [line for line in response.strip().split('\n')[-5:] if line.strip()]
            
            # Check for a position on its own line at the end (highest priority)
            for line in last_non_empty_lines:
                line = line.strip().upper()
                for position in position_names:
                    if line == position or line == position.replace("_", " "):
                        chosen_position = position
                        print_green(f"Found position '{position}' as standalone line at end")
                        break
                if chosen_position:
                    break
            
            # If not found, try other methods
            if not chosen_position:
                # Look for positions in conclusion phrases
                conclusion_phrases = [
                    "SELECTED POSITION", "I SELECT", "I CHOOSE", "MY POSITION", "SELECTED MOVE",
                    "BEST MOVE", "FINAL CHOICE", "I WILL PLACE", "I RECOMMEND", "RECOMMEND PLACING"
                ]
                
                for line in response.split('\n'):
                    for phrase in conclusion_phrases:
                        if phrase in line.upper():
                            for position in position_names:
                                if position in line.upper() or position.replace("_", " ") in line.upper():
                                    chosen_position = position
                                    print_green(f"Found position '{position}' in conclusion: '{line}'")
                                    break
                    if chosen_position:
                        break
                        
            # If still not found, default to CENTER
            if not chosen_position:
                print_yellow("Could not extract a position from Claude's response")
                print_yellow("Using CENTER as fallback")
                chosen_position = "CENTER"
                
            print_green(f"Claude selected position: {chosen_position}")
            return True, chosen_position, situation, reasoning
            
        except Exception as e:
            print_yellow(f"Error in board analysis: {e}")
            return False, None, "", ""
            
    def check_move_completion(self, position):
        """
        Check if the move has been completed successfully by analyzing the board state
        Returns: True if move is complete, False if it needs more time
        """
        print_blue("Checking if move to position {} is complete...".format(position))

        # Get the current board state
        current_img = self.robot.get_current_img()
        if current_img is None:
            print_yellow("Could not get current image for completion check")
            return False

        # Use Claude to analyze if the move was completed
        try:
            # Convert image from BGR to RGB
            img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)

            # Convert image to base64
            success, encoded_img = cv2.imencode(".jpg", img_rgb)
            if not success:
                raise ValueError("Could not encode image")
            img_base64 = base64.b64encode(encoded_img).decode('utf-8')

            # Simplified prompt focused on checking if our move is complete
            check_prompt = """
            IMPORTANT: Look at this tic-tac-toe board and answer with EXTREME CLARITY:

            Question 1: Is there an orange circle (O) placed at the {} position? Answer ONLY "Yes" or "No".
            Question 2: Has the robot successfully completed its move? Answer ONLY "Yes" or "No".

            Format your response as:
            1. [Yes/No]
            2. [Yes/No]

            Then you may provide a ONE SENTENCE explanation if needed.
            """.format(position)

            message = [
                {"type": "text", "text": check_prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}}
            ]

            # Call Claude API with shorter context
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=150,  # Short response
                temperature=0.0,
                messages=[
                    {"role": "user", "content": message}
                ]
            ).content[0].text

            # Debug - always show full response
            print_blue("Completion check response:")
            print(response)

            # Check for negative phrases first (more reliable)
            response_upper = response.upper()
            negative_phrases = ["NO CIRCLE", "NO ORANGE CIRCLE", "NOT COMPLETE",
                             "NOT PLACED", "HAS NOT", "UNSUCCESSFUL", "NOT SUCCESSFUL"]

            for phrase in negative_phrases:
                if phrase in response_upper:
                    print_yellow("Move definitely NOT complete - detected negative phrase: " + phrase)
                    return False

            # Now check for positive indicators
            success_phrases = ["YES", "SUCCESSFUL", "O IS PLACED", "CIRCLE IS PLACED",
                              "ORANGE CIRCLE IS", "ORANGE CIRCLE AT", "COMPLETED"]

            success_count = 0
            matching_phrases = []

            for phrase in success_phrases:
                if phrase in response_upper:
                    success_count += 1
                    matching_phrases.append(phrase)

            # Only consider complete if multiple positive indicators and no negatives
            if success_count >= 2:
                print_green(f"Move completion detected! Matched phrases: {matching_phrases}")
                return True

            # If there are some positive indicators but not enough, check the first line
            # Claude often starts with a simple Yes/No
            first_line = response.strip().split('\n')[0].strip().upper()
            if first_line.startswith("YES") and "NO" not in first_line:
                print_green("Move completion detected from first line confirmation!")
                return True

            # Also check directly for numbers in the first response (common Claude pattern)
            if response.strip().startswith("1. Yes"):
                print_green("Move completion detected from numbered Yes response!")
                return True

            print_yellow("Move not yet complete, continuing execution...")
            return False

        except Exception as e:
            print_yellow(f"Error checking move completion: {e}")
            return False

    def execute_move(self, position, situation, reasoning):
        """Execute a move on the board with persistent execution until completion"""
        with self.lock:
            try:
                print_green(f"Executing move: {position}")

                # Convert position string to enum
                try:
                    task = TaskToString[position]
                except KeyError:
                    print_yellow(f"Invalid position: {position}")
                    return False

                # Set the language instruction for the GR00T client
                self.client.set_lang_instruction(str(task))

                # Create a flag for stopping execution
                self.stop_execution = False

                # Maximum number of execution attempts
                max_attempts = 200  # Increased from 5 to 10
                attempt = 0
                move_completed = False
                total_execution_time = 0  # Track total time spent for this move
                max_total_time = 300  # Maximum 5 minutes total

                # Keep trying until the move is completed, max attempts reached, or total time exceeded
                while (not move_completed and
                       attempt < max_attempts and
                       not self.stop_execution and
                       total_execution_time < max_total_time):
                    attempt += 1
                    print_green(f"Execution attempt {attempt}/{max_attempts}")

                    for i in range(20):
                        # Get the current image and state
                        img = self.robot.get_current_img()
                        state = self.robot.get_current_state()

                        # Get the action from GR00T
                        action = self.client.get_action(img, state)

                        # Execute the action
                        print_green("Executing movement sequence...")

                        # Execute the full action horizon
                        for j in range(ACTION_HORIZON):
                            if self.stop_execution:
                                break

                            concat_action = np.concatenate(
                                [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                                axis=0,
                            )
                            assert concat_action.shape == (6,), f"Expected shape (6,), got {concat_action.shape}"
                            self.robot.set_target_state(torch.from_numpy(concat_action))
                            time.sleep(0.02)  # Same timing as in tictac_bot.py

                            # Show the current image with correct color ordering
                            current_img = self.robot.get_current_img()
                            if current_img is not None:
                                # Convert from BGR to RGB for correct color display
                                display_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                                cv2.imshow("Current Board State", display_img)
                                cv2.waitKey(1)

                    # Check if the move is complete
                    start_check = time.time()
                    move_completed = self.check_move_completion(position)
                    check_duration = time.time() - start_check
                    total_execution_time += check_duration

                    if not move_completed:
                        print_yellow(f"Move to {position} not yet complete, continuing execution...")
                        print_blue(f"Total execution time so far: {total_execution_time:.1f}s / {max_total_time}s")
                    else:
                        print_green(f"Move to {position} successfully completed!")
                        print_green(f"Completed in {total_execution_time:.1f} seconds and {attempt} attempts")

                # Only go home if the move was completed or we reached max attempts
                if move_completed or attempt >= max_attempts:
                    # Move back to home position and wait for the next turn
                    print_blue("Moving back to home position...")
                    self.robot.go_home()
                    time.sleep(1.0)

                # Update last move time
                self.last_move_time = time.time()
                self.moves_made += 1

                print_green(f"Move operation complete. Total moves made: {self.moves_made}")
                return move_completed

            except Exception as e:
                print_yellow(f"Error executing move: {e}")
                return False
                
    def monitor_board(self):
        """Continuously monitor the board for changes and make moves when appropriate"""
        print_green("Starting board monitoring thread...")
        
        while self.is_running:
            # Process any commands in the queue
            while not self.command_queue.empty():
                try:
                    cmd, *args = self.command_queue.get_nowait()
                    if cmd == "PAUSE":
                        self.paused = True
                        print_green("Agent paused - waiting for resume command")
                        self.robot.go_home()
                    elif cmd == "RESUME":
                        self.paused = False
                        print_green("Agent resumed - monitoring board for changes")
                    elif cmd == "STOP":
                        print_green("Stopping monitoring thread...")
                        return
                except queue.Empty:
                    break
            
            # Skip monitoring when paused
            if self.paused:
                time.sleep(0.5)
                continue
                
            # Get current board state
            current_state = self.get_board_state()
            
            if current_state is not None:
                # Analyze the board
                should_move, position, situation, reasoning = self.analyze_board(current_state)
                
                if should_move:
                    # Execute the move
                    success = self.execute_move(position, situation, reasoning)
                    if success:
                        print_green("Move executed successfully")
                        # Update last board state
                        self.last_board_state = self.get_board_state()
                    else:
                        print_yellow("Move execution failed")
                else:
                    print_blue("No move needed at this time - waiting...")
                    
            # Wait before the next check
            time.sleep(self.monitoring_interval)
            
    def start(self):
        """Start the agent"""
        with self.lock:
            if self.is_running:
                print_yellow("Agent is already running")
                return
                
            # Initialize the robot
            self.initialize_robot()
            
            # Start the monitoring thread
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self.monitor_board)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            print_green("Agent started in paused state. Press 'r' to resume monitoring.")
            
    def stop(self):
        """Stop the agent"""
        with self.lock:
            if not self.is_running:
                print_yellow("Agent is not running")
                return
                
            # Signal the monitoring thread to stop
            self.command_queue.put(("STOP",))
            self.is_running = False
            
            # Wait for the thread to finish
            if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
                
            # Clean up resources
            self.cleanup()
            
            print_green("Agent stopped")
            
    def pause(self):
        """Pause the agent"""
        self.command_queue.put(("PAUSE",))
        
    def resume(self):
        """Resume the agent"""
        self.command_queue.put(("RESUME",))
        

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Continuous Agent")
    parser.add_argument("--host", type=str, default=HOST, help=f"VLA server host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"VLA server port (default: {PORT})")
    parser.add_argument("--cam-idx", type=int, default=CAM_IDX, help=f"Camera index (default: {CAM_IDX})")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="Claude API key")
    parser.add_argument("--model", type=str, default=CLAUDE_MODEL, help=f"Claude model (default: {CLAUDE_MODEL})")
    parser.add_argument("--interval", type=float, default=MONITORING_INTERVAL, 
                        help=f"Monitoring interval in seconds (default: {MONITORING_INTERVAL})")
    args = parser.parse_args()
    
    # Create the agent
    try:
        agent = TicTacToeAgent(
            api_key=args.api_key,
            host=args.host,
            port=args.port,
            cam_idx=args.cam_idx,
            claude_model=args.model,
            monitoring_interval=args.interval
        )
        
        # Start the agent
        agent.start()
        
        # Set up keyboard control
        print_green("==== Tic-Tac-Toe Continuous Agent ====")
        print_green("Press 'r' to resume monitoring")
        print_green("Press 'p' to pause the agent")
        print_green("Press 'q' to quit")
        
        # Main control loop
        from pynput import keyboard
        
        def on_key_press(key):
            try:
                if key.char == 'q':
                    print_green("Quitting...")
                    agent.stop()
                    return False  # Stop the listener
                elif key.char == 'p':
                    print_green("Pausing agent...")
                    agent.pause()
                elif key.char == 'r':
                    print_green("Resuming agent...")
                    agent.resume()
            except AttributeError:
                pass
                
        # Start keyboard listener
        with keyboard.Listener(on_press=on_key_press) as listener:
            listener.join()
            
    except KeyboardInterrupt:
        print_green("\nStopping agent...")
        if 'agent' in locals():
            agent.stop()
        print_green("Exiting...")
        
    except Exception as e:
        print_yellow(f"Error: {e}")
        if 'agent' in locals():
            agent.stop()
            
    finally:
        # Make sure to close OpenCV windows
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()