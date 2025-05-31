#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MCP Server for Tic-Tac-Toe Robot Control

This script implements a local MCP (Model-Connector-Planner) server that allows 
Claude to interact with the GR00T VLA (Vision-Language-Action) model to play tic-tac-toe.

This final version fixes camera allocation and resource management issues.
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import cv2
import flask
import numpy as np
import torch
from eval_gr00t_so100 import Gr00tRobotInferenceClient, SO100Robot, view_img
from flask import Flask, jsonify, request
from flask_cors import CORS


# Configuration constants
ACTION_HORIZON = 16
ACTIONS_TO_EXECUTE = 10
MODALITY_KEYS = ["single_arm", "gripper"]
HOST = "localhost"  # The VLA server IP address
PORT = 5555  # The VLA server port
CAM_IDX = 8  # The camera index


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


def fix_image_for_display(img):
    """
    Prepare an image for display with OpenCV:
    1. Convert from BGR to RGB for proper color display
    2. Return the fixed image
    """
    if img is None:
        return None

    try:
        # OpenCV uses BGR format by default, convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print_yellow(f"Error converting image colors: {e}")
        return img  # Return original if conversion fails


def find_available_cameras():
    """Find all available camera indices"""
    available_indices = []
    
    # On Linux systems, scan for video devices
    try:
        import os
        if os.path.exists("/dev"):
            video_devices = sorted([int(f.replace("video", "")) for f in os.listdir("/dev") 
                           if f.startswith("video") and f[5:].isdigit()])
            
            if video_devices:
                print_green(f"Found video devices: {video_devices}")
                for idx in video_devices:
                    try:
                        cap = cv2.VideoCapture(idx)
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            available_indices.append(idx)
                            print_green(f"Verified working camera at index {idx}")
                    except Exception as e:
                        print_yellow(f"Device {idx} exists but had error: {e}")
    except Exception as e:
        print_yellow(f"Error scanning for video devices: {e}")
    
    # If no cameras found yet, try common indices
    if not available_indices:
        for idx in [0, 1, 2, 4, 6, 8, 10]:
            try:
                cap = cv2.VideoCapture(idx)
                ret, _ = cap.read()
                cap.release()
                if ret:
                    available_indices.append(idx)
                    print_green(f"Found working camera at index {idx}")
            except:
                pass
    
    return available_indices


class RobotController:
    """
    Manages the robot and GR00T VLA client with proper resource management.
    Only initializes the robot once and maintains the connection.
    """
    
    def __init__(self, host=HOST, port=PORT, cam_idx=CAM_IDX):
        self.host = host
        self.port = port
        self.cam_idx = cam_idx
        self.robot = None
        self.client = None
        self.is_initialized = False
        self.is_executing = False
        self.last_image = None
        self.lock = threading.RLock()
        
        # Find available cameras at initialization
        self.available_cameras = find_available_cameras()
        if self.available_cameras:
            print_green(f"Available cameras: {self.available_cameras}")
            # Prioritize the requested camera if available
            if self.cam_idx in self.available_cameras:
                print_green(f"Requested camera index {self.cam_idx} is available")
            else:
                print_yellow(f"Requested camera index {self.cam_idx} not found in available cameras")
                if self.available_cameras:
                    self.cam_idx = self.available_cameras[0]
                    print_green(f"Using first available camera index: {self.cam_idx}")
        else:
            print_yellow(f"No cameras found! Will try with index {self.cam_idx} anyway")
        
        # Initialize immediately - this is crucial for proper resource management
        self.initialize()
        
    def initialize(self):
        """Initialize the robot and GR00T client only once"""
        with self.lock:
            if self.is_initialized:
                return True
                
            try:
                # Initialize the GR00T client
                initial_task = TaskToString.CENTER  # Default task
                self.client = Gr00tRobotInferenceClient(
                    host=self.host,
                    port=self.port,
                    language_instruction=str(initial_task),
                )
                
                # Initialize the robot with the working camera
                success = False
                
                # First try with the requested camera
                try:
                    print_green(f"Initializing robot with camera index {self.cam_idx}")
                    self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=self.cam_idx)
                    self.robot.connect()
                    self.robot.move_to_initial_pose()
                    success = True
                except Exception as e:
                    print_yellow(f"Error initializing with camera {self.cam_idx}: {e}")
                    # Try alternate cameras only if the primary fails
                    for idx in self.available_cameras:
                        if idx == self.cam_idx:
                            continue  # Skip the one we already tried
                        
                        try:
                            print_green(f"Trying with alternate camera index {idx}")
                            self.cam_idx = idx
                            self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=idx)
                            self.robot.connect()
                            self.robot.move_to_initial_pose()
                            success = True
                            break
                        except Exception as e2:
                            print_yellow(f"Error with camera {idx}: {e2}")
                
                if not success:
                    raise ValueError("Could not initialize with any available camera")
                
                self.is_initialized = True
                print_green("Robot and GR00T client successfully initialized")
                return True
            except Exception as e:
                print_yellow(f"Failed to initialize robot: {e}")
                self.cleanup()
                return False
            
    def cleanup(self):
        """Clean up resources - only called on shutdown"""
        with self.lock:
            if not self.is_initialized:
                return
                
            try:
                if self.robot:
                    self.robot.disconnect()
                self.is_initialized = False
                print_green("Robot disconnected")
            except Exception as e:
                print_yellow(f"Error during cleanup: {e}")
            
    def get_current_image(self):
        """Get the current camera image from the robot"""
        with self.lock:
            if not self.is_initialized:
                print_yellow("Robot not initialized when getting image")
                return None

            try:
                # Use the get_current_img method from SO100Robot
                img = self.robot.get_current_img()

                # Log image details for debugging
                if img is not None:
                    print_green(f"Got image from camera: shape={img.shape}, dtype={img.dtype}")

                    # Save in cache for future use if needed
                    self.last_image = img

                    # The image is kept in BGR format here since OpenCV expects it
                    # We'll convert to RGB only before display or encoding
                    return img
                else:
                    print_yellow("Camera returned None image")
                    return None
            except Exception as e:
                print_yellow(f"Error getting image: {e}")
                try:
                    # Print available observations for debugging
                    obs_keys = self.robot.get_observation().keys()
                    print_yellow(f"Available observation keys: {obs_keys}")
                except:
                    print_yellow("Could not get observation keys")
                return None
            
    def execute_move(self, position: str):
        """Execute a move based on the position name, continuing until user presses Enter"""
        import threading

        with self.lock:
            if not self.is_initialized:
                print_yellow("Robot not initialized when executing move")
                return False

            if self.is_executing:
                print_yellow("Already executing a move. Please wait.")
                return False

            try:
                self.is_executing = True

                # Convert position string to enum
                try:
                    task = TaskToString[position.upper()]
                except KeyError:
                    print_yellow(f"Invalid position: {position}")
                    self.is_executing = False
                    return False

                print_green(f"Executing move: {task}")
                print_green("Press Enter when you want to stop the execution...")

                # Set the language instruction for the GR00T client
                self.client.set_lang_instruction(str(task))

                # Flag to control execution
                should_stop = False

                # Function to check for Enter key press
                def check_for_enter():
                    nonlocal should_stop
                    input()  # Wait for Enter key
                    should_stop = True
                    print_green("Stopping execution after current action...")

                # Start thread to monitor for Enter key
                monitor_thread = threading.Thread(target=check_for_enter)
                monitor_thread.daemon = True
                monitor_thread.start()

                print_green("Beginning execution. Press Enter to stop...")

                # Main control loop - directly adapted from tictac_bot.py
                while not should_stop:
                    # Get current image and display it
                    img = self.robot.get_current_img()
                    if img is not None:
                        cv2.namedWindow("Robot Camera", cv2.WINDOW_NORMAL)
                        cv2.imshow("Robot Camera", img)
                        cv2.waitKey(1)  # Update the display without blocking

                    # Get current state and new action from GR00T
                    state = self.robot.get_current_state()
                    action = self.client.get_action(img, state)

                    # Execute the action horizon
                    for j in range(ACTION_HORIZON):
                        if should_stop:  # Check if we should stop during action
                            break

                        # Build and execute the action
                        concat_action = np.concatenate(
                            [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                            axis=0,
                        )
                        assert concat_action.shape == (6,), f"Expected shape (6,), got {concat_action.shape}"
                        self.robot.set_target_state(torch.from_numpy(concat_action))
                        time.sleep(0.02)  # Same timing as in tictac_bot.py

                        # Get and display the real-time image
                        img = self.robot.get_current_img()
                        if img is not None:
                            cv2.imshow("Robot Camera", img)
                            cv2.waitKey(1)

                # Wait for the monitor thread to finish if it hasn't already
                if monitor_thread.is_alive():
                    monitor_thread.join(timeout=1.0)

                print_green("Execution stopped by user")

                # Go to home position
                self.robot.go_home()
                time.sleep(1.0)

                self.is_executing = False
                return True

            except Exception as e:
                print_yellow(f"Error executing move: {e}")
                self.is_executing = False
                return False
            
    def move_home(self):
        """Move the robot directly to home position"""
        with self.lock:
            if not self.is_initialized:
                print_yellow("Robot not initialized when moving home")
                return False

            try:
                print_green("Moving robot to home position...")

                # Go directly to home position without first going to initial pose
                self.robot.go_home()
                time.sleep(3.0)  # Wait for home position movement

                print_green("Robot successfully moved to home position")
                return True
            except Exception as e:
                print_yellow(f"Error moving to home: {e}")
                return False


# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a single global robot controller instance
# This is crucial for proper resource management
controller = None


# Thread lock for controller initialization
controller_init_lock = threading.Lock()

# Using Flask's before_request handler instead of deprecated before_first_request
@app.before_request
def initialize_controller_if_needed():
    """Initialize the controller if it hasn't been initialized yet (thread-safe)"""
    global controller
    if controller is None:
        with controller_init_lock:
            # Check again inside the lock to prevent concurrent initialization
            if controller is None:
                print_green("Initializing robot controller for the first time")
                controller = RobotController(
                    host=app.config['VLA_HOST'],
                    port=app.config['VLA_PORT'],
                    cam_idx=app.config['CAM_IDX']
                )


@app.route('/api/reset', methods=['POST'])
def reset_robot():
    """Reset the robot to its home position"""
    global controller
    print_green("Resetting robot to home position...")

    # Try to get a fresh image before reset
    try:
        if controller and controller.robot and hasattr(controller.robot, 'get_current_img'):
            img_before = controller.robot.get_current_img()
            if img_before is not None:
                # Display the image before reset
                cv2.namedWindow("Before Reset", cv2.WINDOW_NORMAL)
                cv2.imshow("Before Reset", fix_image_for_display(img_before))
                cv2.waitKey(1)
    except Exception as e:
        print_yellow(f"Error showing pre-reset image: {e}")

    # Reset the robot
    success = controller.move_home()

    # Try to get a fresh image after reset
    try:
        if controller and controller.robot and hasattr(controller.robot, 'get_current_img'):
            img_after = controller.robot.get_current_img()
            if img_after is not None:
                # Display the image after reset
                cv2.namedWindow("After Reset", cv2.WINDOW_NORMAL)
                cv2.imshow("After Reset", fix_image_for_display(img_after))
                cv2.waitKey(1000)
    except Exception as e:
        print_yellow(f"Error showing post-reset image: {e}")

    return jsonify({
        "success": success,
        "message": "Robot reset to home position" if success else "Failed to reset robot"
    })


@app.route('/api/get_board_state', methods=['GET'])
def get_board_state():
    """Get the current board state (image)"""
    global controller

    # Force getting a fresh image directly from the camera
    if controller and controller.robot and hasattr(controller.robot, 'get_current_img'):
        try:
            # Try to capture a fresh image directly
            img = controller.robot.get_current_img()
            print_green("Successfully captured fresh image from camera")
        except Exception as e:
            print_yellow(f"Error getting fresh image directly: {e}")
            # Fall back to the controller method
            img = controller.get_current_image()
    else:
        img = controller.get_current_image()

    if img is None:
        return jsonify({
            "success": False,
            "message": "Failed to capture image"
        }), 500

    # Debug info about the image
    print_green(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # Display the current board state in a window - update every time
    print_green("Refreshed board state:")
    try:
        # Create a resized copy for display, maintaining aspect ratio
        height, width = img.shape[:2]
        max_dimension = 640
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))

        display_img = cv2.resize(img, (new_width, new_height))

        # Make sure window is always on top and refreshed
        cv2.namedWindow("Current Board State", cv2.WINDOW_NORMAL)
        cv2.imshow("Current Board State", fix_image_for_display(display_img))
        cv2.waitKey(1)  # Show without blocking
    except Exception as e:
        print_yellow(f"Error displaying image: {e}")

    # Convert the image to base64
    try:
        # Convert to RGB before encoding for transmission to Claude
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        success, encoded_img = cv2.imencode('.jpg', img_rgb)
        if not success:
            return jsonify({
                "success": False,
                "message": "Failed to encode image"
            }), 500

        img_base64 = base64.b64encode(encoded_img).decode('utf-8')
        print_green("Successfully encoded RGB image for transmission")

        return jsonify({
            "success": True,
            "image": img_base64,
            "timestamp": time.time(),
            "description_prompt": """
Please analyze this tic-tac-toe board image carefully and provide:

1. SITUATION DESCRIPTION:
   - Describe the current board state in detail
   - Which positions are occupied by Xs (human)
   - Which positions are occupied by Os (robot/circles)
   - Which positions are empty
   - Label positions using this reference:
     TOP_LEFT | CENTER_TOP | TOP_RIGHT
     ---------+------------+----------
     CENTER_LEFT | CENTER | CENTER_RIGHT
     ---------+------------+----------
     BOTTOM_LEFT | CENTER_BOTTOM | BOTTOM_RIGHT

2. GAME ANALYSIS:
   - Who played last?
   - Whose turn is next?
   - Is there a winner? If so, who?
   - Is the game a draw?

3. REASONING:
   - Based on this analysis, what's the best next move?
   - Explain your reasoning strategically

After your analysis, provide your selected position using one of these values exactly:
TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER, CENTER_TOP, CENTER_BOTTOM, CENTER_LEFT, CENTER_RIGHT
"""
        })
    except Exception as e:
        print_yellow(f"Error encoding image: {e}")
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500


@app.route('/api/get_available_positions', methods=['GET'])
def get_available_positions():
    """Get all available positions on the board"""
    positions = []
    for position in TaskToString:
        positions.append({
            "name": position.name,
            "description": position.value
        })
        
    return jsonify({
        "success": True,
        "positions": positions
    })


@app.route('/api/execute_move', methods=['POST'])
def execute_move():
    """Execute a move on the board"""
    global controller

    data = request.json
    if not data or 'position' not in data:
        return jsonify({
            "success": False,
            "message": "Position not specified"
        }), 400

    # Extract data provided by Claude
    position = data['position']
    reasoning = data.get('reasoning', '')
    situation = data.get('situation', '')

    # Display situation and reasoning in terminal with more visual emphasis
    print("\n" + "="*100)
    print_green("🤖 CLAUDE'S ANALYSIS AND PLANNING:")
    print_green("="*100)

    # Check if Claude provided detailed analysis
    if not situation or len(situation) < 20:  # If missing or too short
        print_yellow("Claude did not provide a detailed situation analysis.")
        print_yellow("Make sure you're passing the description_prompt to Claude and asking it to analyze the board state.")
        situation = f"I see the tic-tac-toe board and will place a marker at position: {position}"

    if not reasoning or len(reasoning) < 20:  # If missing or too short
        print_yellow("Claude did not provide detailed reasoning.")
        print_yellow("Make sure you're asking Claude to explain its reasoning for the move.")

        # Generate a default reasoning based on the position
        position_map = {
            "CENTER": "The center position is strategically strong as it controls the most lines.",
            "TOP_LEFT": "The top-left corner gives control of two potential lines.",
            "TOP_RIGHT": "The top-right corner gives control of two potential lines.",
            "BOTTOM_LEFT": "The bottom-left corner gives control of two potential lines.",
            "BOTTOM_RIGHT": "The bottom-right corner gives control of two potential lines.",
            "CENTER_TOP": "The center-top position controls the top row and middle column.",
            "CENTER_BOTTOM": "The center-bottom position controls the bottom row and middle column.",
            "CENTER_LEFT": "The center-left position controls the left column and middle row.",
            "CENTER_RIGHT": "The center-right position controls the right column and middle row."
        }
        reasoning = position_map.get(position, f"This position seems to be the best move given the current board state.")

    # Print with additional formatting
    print_yellow("📷 WHAT I SEE:")
    print(f"{situation}\n")

    print_yellow("🤔 MY REASONING:")
    print(f"{reasoning}\n")

    print_yellow("🎯 SELECTED POSITION:")
    print_green(f"{position}")
    print("="*100 + "\n")

    # Get current board state before move
    # Try to get directly from the robot's camera
    if controller and controller.robot and hasattr(controller.robot, 'get_current_img'):
        try:
            img_before = controller.robot.get_current_img()
        except Exception as e:
            print_yellow(f"Error getting direct image: {e}")
            img_before = controller.get_current_image()
    else:
        img_before = controller.get_current_image()

    if img_before is not None:
        # Display the image before move
        print_green("Current board state (before move):")
        try:
            # Create a resized copy for display, maintaining aspect ratio
            height, width = img_before.shape[:2]
            max_dimension = 640
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))

            display_img = cv2.resize(img_before, (new_width, new_height))

            # Make sure window is always on top and refreshed
            cv2.namedWindow("Board State (Before Move)", cv2.WINDOW_NORMAL)
            cv2.imshow("Board State (Before Move)", fix_image_for_display(display_img))
            cv2.waitKey(1)  # Show without blocking
        except Exception as e:
            print_yellow(f"Error displaying before image: {e}")

    # Execute the move
    success = controller.execute_move(position)

    # After executing the move, get a new image - try direct camera access first
    if controller and controller.robot and hasattr(controller.robot, 'get_current_img'):
        try:
            img_after = controller.robot.get_current_img()
        except Exception as e:
            print_yellow(f"Error getting direct after image: {e}")
            img_after = controller.get_current_image()
    else:
        img_after = controller.get_current_image()

    img_base64 = None
    if img_after is not None:
        # Display the image after move
        print_green("Updated board state (after move):")
        try:
            # Create a resized copy for display, maintaining aspect ratio
            height, width = img_after.shape[:2]
            max_dimension = 640
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))

            display_img = cv2.resize(img_after, (new_width, new_height))

            # Make sure window is always on top and refreshed
            cv2.namedWindow("Board State (After Move)", cv2.WINDOW_NORMAL)
            cv2.imshow("Board State (After Move)", fix_image_for_display(display_img))
            cv2.waitKey(1000)  # Show for 1 second
        except Exception as e:
            print_yellow(f"Error displaying after image: {e}")

        try:
            # Convert to RGB before encoding for transmission to Claude
            img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
            success, encoded_img = cv2.imencode('.jpg', img_after_rgb)
            if success:
                img_base64 = base64.b64encode(encoded_img).decode('utf-8')
                print_green("Successfully encoded RGB image for transmission")
        except Exception as e:
            print_yellow(f"Error encoding after image: {e}")

    return jsonify({
        "success": success,
        "message": f"Successfully executed move at {position}" if success else f"Failed to execute move at {position}",
        "image": img_base64
    })


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get information about the server and available endpoints"""
    global controller
    
    return jsonify({
        "name": "Tic-Tac-Toe MCP Server",
        "version": "1.0.0",
        "camera_index": controller.cam_idx if controller else None,
        "available_cameras": controller.available_cameras if controller else [],
        "robot_initialized": controller.is_initialized if controller else False,
        "endpoints": [
            {"path": "/api/reset", "method": "POST", "description": "Reset the robot to home position"},
            {"path": "/api/get_board_state", "method": "GET", "description": "Get the current board state (image)"},
            {"path": "/api/get_available_positions", "method": "GET", "description": "Get all available positions on the board"},
            {"path": "/api/execute_move", "method": "POST", "description": "Execute a move on the board", 
             "parameters": {"position": "The position name (e.g., 'CENTER', 'TOP_LEFT')"}},
            {"path": "/api/info", "method": "GET", "description": "Get server information"}
        ]
    })


@app.route('/', methods=['GET'])
def index():
    """Serve a simple HTML page with information about the API"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tic-Tac-Toe MCP Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .get { background-color: #61affe; color: white; }
            .post { background-color: #49cc90; color: white; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .live-img { margin-top: 20px; text-align: center; }
            img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; 
                     border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background-color: #45a049; }
            .status { padding: 10px; margin-bottom: 20px; border-radius: 5px; background-color: #f8f9fa; }
            .status.ok { background-color: #d4edda; color: #155724; }
            .status.error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>Tic-Tac-Toe MCP Server</h1>
        <p>This server provides API endpoints for controlling the Tic-Tac-Toe robot using Claude via MCP.</p>
        
        <div id="status" class="status">Checking server status...</div>
        
        <h2>API Endpoints</h2>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/info</code>
            <p>Get information about the server and available endpoints.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/get_board_state</code>
            <p>Get the current board state (image).</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/get_available_positions</code>
            <p>Get all available positions on the board.</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/execute_move</code>
            <p>Execute a move on the board.</p>
            <p>Request body:</p>
            <pre>{"position": "CENTER"}</pre>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/reset</code>
            <p>Reset the robot to its home position.</p>
        </div>
        
        <div class="live-img">
            <h2>Live Board View</h2>
            <img id="board-img" src="" alt="Board not available">
            <div>
                <button onclick="refreshImage()">Refresh Image</button>
                <button onclick="resetRobot()">Reset Robot</button>
            </div>
        </div>
        
        <script>
            // Check server status on load
            function checkStatus() {
                fetch('/api/info')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const initialized = data.robot_initialized;
                        const cameraIndex = data.camera_index;
                        const availableCameras = data.available_cameras.join(', ');
                        
                        if (initialized) {
                            statusDiv.className = 'status ok';
                            statusDiv.innerHTML = `<strong>Server Status:</strong> Ready | Camera: ${cameraIndex} | Available cameras: [${availableCameras}]`;
                        } else {
                            statusDiv.className = 'status error';
                            statusDiv.innerHTML = `<strong>Server Status:</strong> Robot not initialized | Available cameras: [${availableCameras}]`;
                        }
                    })
                    .catch(error => {
                        console.error('Error checking status:', error);
                        const statusDiv = document.getElementById('status');
                        statusDiv.className = 'status error';
                        statusDiv.innerHTML = '<strong>Server Status:</strong> Error connecting to server';
                    });
            }
            
            function refreshImage() {
                fetch('/api/get_board_state')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('board-img').src = 'data:image/jpeg;base64,' + data.image;
                        } else {
                            alert('Failed to get image: ' + data.message);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error fetching image');
                    });
            }
            
            function resetRobot() {
                fetch('/api/reset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    refreshImage();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error resetting robot');
                });
            }
            
            // Initialize on page load
            window.onload = function() {
                checkStatus();
                refreshImage();
                
                // Refresh status every 10 seconds
                setInterval(checkStatus, 10000);
            };
        </script>
    </body>
    </html>
    """
    return html


# Cleanup handler
@app.teardown_appcontext
def teardown_appcontext(exception=None):
    """This no longer disconnects the robot on every request"""
    pass


# Proper cleanup on shutdown
def cleanup_on_shutdown():
    """Clean up resources on server shutdown"""
    global controller
    if controller:
        print_green("Cleaning up robot controller on shutdown")
        controller.cleanup()

    # Close all OpenCV windows
    print_green("Closing all display windows")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--vla-host", type=str, default=HOST, help=f"VLA server host (default: {HOST})")
    parser.add_argument("--vla-port", type=int, default=PORT, help=f"VLA server port (default: {PORT})")
    parser.add_argument("--cam-idx", type=int, default=CAM_IDX, help=f"Camera index (default: {CAM_IDX})")
    args = parser.parse_args()
    
    # Store configuration in app
    app.config['VLA_HOST'] = args.vla_host
    app.config['VLA_PORT'] = args.vla_port
    app.config['CAM_IDX'] = args.cam_idx
    
    # Print startup information
    print_green(f"Starting Tic-Tac-Toe MCP Server on http://{args.host}:{args.port}")
    print_green(f"Will connect to VLA server at {args.vla_host}:{args.vla_port}")
    print_green(f"Using camera index: {args.cam_idx}")
    print_green("Press Ctrl+C to stop the server")
    
    # Initialize controller before running the app - this is the key improvement
    # Note: This will be the primary initialization path, making the before_request handler a fallback
    if controller is None:
        controller = RobotController(
            host=args.vla_host,
            port=args.vla_port,
            cam_idx=args.cam_idx
        )
    
    try:
        # Register the cleanup function to be called on exit
        import atexit
        atexit.register(cleanup_on_shutdown)
        
        # Run the Flask app
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print_green("\nShutting down server...")
        cleanup_on_shutdown()