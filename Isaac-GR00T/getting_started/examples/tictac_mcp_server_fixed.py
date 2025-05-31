#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MCP Server for Tic-Tac-Toe Robot Control

This script implements a local MCP (Model-Connector-Planner) server that allows 
Claude to interact with the GR00T VLA (Vision-Language-Action) model to play tic-tac-toe.

The server exposes API endpoints that Claude can call to:
1. Get the current state of the game (board image)
2. Get available moves (positions on the board)
3. Execute a move by controlling the robot

This approach allows Claude to use structured API calls instead of traditional prompt engineering.
"""

import argparse
import base64
import io
import json
import os
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


def find_available_cameras():
    """Find all available camera indices"""
    # Try to list cameras directly using OpenCV
    available_indices = []
    
    # First try to get cameras from /dev/video* on Linux systems
    try:
        import os
        if os.path.exists("/dev"):
            video_devices = sorted([int(f.replace("video", "")) for f in os.listdir("/dev") 
                           if f.startswith("video") and f[5:].isdigit()])
            
            if video_devices:
                print_green(f"Found video devices: {video_devices}")
                
                # Test each device with OpenCV to verify it works
                for idx in video_devices:
                    try:
                        cap = cv2.VideoCapture(idx)
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            available_indices.append(idx)
                            print_green(f"Verified working camera at index {idx}")
                    except:
                        pass
    except Exception as e:
        print_yellow(f"Error scanning for video devices: {e}")
    
    # If no cameras found through /dev, try common indices
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
    """Manages the robot and GR00T VLA client"""
    
    def __init__(self, host=HOST, port=PORT, cam_idx=CAM_IDX):
        self.host = host
        self.port = port
        self.cam_idx = cam_idx
        self.robot = None
        self.client = None
        self.is_initialized = False
        self.is_executing = False
        self.last_image = None
        
        # Find available cameras at initialization
        self.available_cameras = find_available_cameras()
        if self.available_cameras:
            print_green(f"Available cameras: {self.available_cameras}")
            # Prioritize the requested camera if available
            if self.cam_idx in self.available_cameras:
                print_green(f"Requested camera index {self.cam_idx} is available")
            else:
                self.cam_idx = self.available_cameras[0]
                print_green(f"Using first available camera index: {self.cam_idx}")
        else:
            print_yellow(f"No cameras found! Will try with index {self.cam_idx} anyway")
        
    def initialize(self):
        """Initialize the robot and GR00T client"""
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
            try:
                print_green(f"Initializing robot with camera index {self.cam_idx}")
                self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=self.cam_idx)
                
                # Connect the robot
                self.robot.connect()
                self.robot.move_to_initial_pose()
            except Exception as e:
                print_yellow(f"Error initializing with camera {self.cam_idx}: {e}")
                
                # Try with each available camera until one works
                success = False
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
        """Clean up resources"""
        try:
            if self.robot:
                self.robot.disconnect()
            self.is_initialized = False
            print_green("Robot disconnected")
        except Exception as e:
            print_yellow(f"Error during cleanup: {e}")
            
    def get_current_image(self):
        """Get the current camera image from the robot"""
        if not self.is_initialized:
            if not self.initialize():
                return None
                
        try:
            # Use the get_current_img method from SO100Robot
            img = self.robot.get_current_img()
            # Note: get_current_img already converts BGR to RGB in SO100Robot class
            self.last_image = img
            return img
        except Exception as e:
            print_yellow(f"Error getting image: {e}")
            try:
                # Print available observations to help with debugging
                obs_keys = self.robot.get_observation().keys()
                print_yellow(f"Available observation keys: {obs_keys}")
                
                # If we have the stationary camera key but it's failing, there might be another issue
                if "observation.images.stationary" in obs_keys:
                    print_yellow("Camera key exists but get_current_img failed")
            except:
                print_yellow("Could not get observation keys")
            return None
            
    def execute_move(self, position: str):
        """Execute a move based on the position name"""
        if not self.is_initialized:
            if not self.initialize():
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
            
            # Set the language instruction for the GR00T client
            self.client.set_lang_instruction(str(task))
            
            # Get the current image and state
            img = self.robot.get_current_img()
            state = self.robot.get_current_state()
            
            # Get the action from GR00T
            action = self.client.get_action(img, state)
            
            # Execute the action
            for j in range(ACTION_HORIZON):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[f"action.{key}"][j]) for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (6,), concat_action.shape
                self.robot.set_target_state(torch.from_numpy(concat_action))
                time.sleep(0.05)  # Slightly longer delay for more stable movements
                
            # Allow some time for the robot to settle
            time.sleep(1.0)
            
            self.is_executing = False
            return True
        except Exception as e:
            print_yellow(f"Error executing move: {e}")
            self.is_executing = False
            return False
            
    def move_home(self):
        """Move the robot to home position"""
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        try:
            self.robot.go_home()
            time.sleep(1.0)
            return True
        except Exception as e:
            print_yellow(f"Error moving to home: {e}")
            return False


# Create Flask app and robot controller
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
controller = RobotController()


@app.route('/api/reset', methods=['POST'])
def reset_robot():
    """Reset the robot to its home position"""
    success = controller.move_home()
    return jsonify({
        "success": success,
        "message": "Robot reset to home position" if success else "Failed to reset robot"
    })


@app.route('/api/get_board_state', methods=['GET'])
def get_board_state():
    """Get the current board state (image)"""
    img = controller.get_current_image()
    if img is None:
        return jsonify({
            "success": False,
            "message": "Failed to capture image"
        }), 500
        
    # Convert the image to base64
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        return jsonify({
            "success": False,
            "message": "Failed to encode image"
        }), 500
        
    img_base64 = base64.b64encode(encoded_img).decode('utf-8')
    
    return jsonify({
        "success": True,
        "image": img_base64,
        "timestamp": time.time()
    })


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
    data = request.json
    if not data or 'position' not in data:
        return jsonify({
            "success": False,
            "message": "Position not specified"
        }), 400
        
    position = data['position']
    success = controller.execute_move(position)
    
    # After executing the move, get a new image
    img = controller.get_current_image()
    img_base64 = None
    if img is not None:
        success, encoded_img = cv2.imencode('.jpg', img)
        if success:
            img_base64 = base64.b64encode(encoded_img).decode('utf-8')
    
    return jsonify({
        "success": success,
        "message": f"Successfully executed move at {position}" if success else f"Failed to execute move at {position}",
        "image": img_base64
    })


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get information about the server and available endpoints"""
    return jsonify({
        "name": "Tic-Tac-Toe MCP Server",
        "version": "1.0.0",
        "camera_index": controller.cam_idx,
        "available_cameras": controller.available_cameras,
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
        </style>
    </head>
    <body>
        <h1>Tic-Tac-Toe MCP Server</h1>
        <p>This server provides API endpoints for controlling the Tic-Tac-Toe robot using Claude via MCP.</p>
        
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
            
            // Refresh image on page load
            window.onload = refreshImage;
        </script>
    </body>
    </html>
    """
    return html


# Cleanup handler
@app.teardown_appcontext
def teardown_appcontext(exception=None):
    controller.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--vla-host", type=str, default=HOST, help=f"VLA server host (default: {HOST})")
    parser.add_argument("--vla-port", type=int, default=PORT, help=f"VLA server port (default: {PORT})")
    parser.add_argument("--cam-idx", type=int, default=CAM_IDX, help=f"Camera index (default: {CAM_IDX})")
    args = parser.parse_args()
    
    # Set up the controller with VLA server details
    controller.host = args.vla_host
    controller.port = args.vla_port
    controller.cam_idx = args.cam_idx
    
    # Print startup information
    print_green(f"Starting Tic-Tac-Toe MCP Server on http://{args.host}:{args.port}")
    print_green(f"Will connect to VLA server at {args.vla_host}:{args.vla_port}")
    print_green(f"Using camera index: {args.cam_idx}")
    print_green("Press Ctrl+C to stop the server")
    
    # Don't initialize immediately - let the first API call trigger initialization
    # This avoids startup errors preventing the server from starting
    
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print_green("\nShutting down server...")
        controller.cleanup()