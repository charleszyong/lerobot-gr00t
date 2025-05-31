#!/usr/bin/env python3
"""
MCP Server for Chess Robot Control
Provides tools for an AI agent to control a physical chess robot
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import numpy as np
from contextlib import redirect_stdout

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import chess robot modules
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from chess_robot.scripts.execute_trajectory import TrajectoryExecutor
from chess_robot.scripts.trajectory_utils import TrajectoryManager

# Configure logging to stderr to avoid stdout pollution
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChessRobotMCPServer:
    """MCP Server for controlling chess robot"""
    
    def __init__(self):
        self.server = Server("chess-robot")
        self.robot: Optional[Robot] = None
        self.executor: Optional[TrajectoryExecutor] = None
        self.manager = TrajectoryManager()
        self.setup_handlers()
        
    def disable_robot_motors(self):
        """Disable torque on all robot motors to prevent overheating"""
        if not self.robot or not isinstance(self.robot, ManipulatorRobot):
            return
        
        try:
            # Import the appropriate TorqueMode based on the robot configuration
            if hasattr(self.robot.config, 'motor_type'):
                if self.robot.config.motor_type == 'feetech':
                    from lerobot.common.robot_devices.motors.feetech import TorqueMode
                else:
                    from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
            else:
                # Default to feetech for SO100
                from lerobot.common.robot_devices.motors.feetech import TorqueMode
            
            # Disable torque on all follower arm motors
            for name in self.robot.follower_arms:
                try:
                    self.robot.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
                    logger.info(f"Disabled torque on follower arm '{name}'")
                except Exception as e:
                    logger.warning(f"Failed to disable torque on follower arm '{name}': {e}")
            
            logger.info("Robot motors disabled to prevent overheating")
        except Exception as e:
            logger.error(f"Error disabling robot motors: {e}")
    
    def setup_handlers(self):
        """Set up MCP tool handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_robot_status",
                    description="Check if the robot is connected and ready",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="initialize_robot",
                    description="Initialize connection to the chess robot",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="disconnect_robot",
                    description="Disconnect from the chess robot",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="execute_chess_move",
                    description="Execute a complete chess move by picking up a piece from one square and putting it down on another",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "from_square": {
                                "type": "string",
                                "description": "Source square (e.g., 'e2')"
                            },
                            "to_square": {
                                "type": "string",
                                "description": "Destination square (e.g., 'e4')"
                            },
                            "speed_factor": {
                                "type": "number",
                                "description": "Speed factor (0.1-2.0, default 1.0)",
                                "default": 1.0
                            }
                        },
                        "required": ["from_square", "to_square"]
                    }
                ),
                Tool(
                    name="execute_single_trajectory",
                    description="Execute a single trajectory (pickup or putdown) for a specific square",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "square": {
                                "type": "string",
                                "description": "Chess square (e.g., 'e4')"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["pickup", "putdown"],
                                "description": "Action type"
                            },
                            "speed_factor": {
                                "type": "number",
                                "description": "Speed factor (0.1-2.0, default 1.0)",
                                "default": 1.0
                            }
                        },
                        "required": ["square", "action"]
                    }
                ),
                Tool(
                    name="list_available_trajectories",
                    description="List all recorded chess trajectories (pickup and putdown positions)",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_trajectory_info",
                    description="Get information about a specific trajectory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "square": {
                                "type": "string",
                                "description": "Chess square (e.g., 'e4')"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["pickup", "putdown"],
                                "description": "Action type"
                            }
                        },
                        "required": ["square", "action"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "get_robot_status":
                    status = await self.get_robot_status()
                    return [TextContent(type="text", text=json.dumps(status, indent=2))]
                
                elif name == "initialize_robot":
                    result = await self.initialize_robot()
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "disconnect_robot":
                    result = await self.disconnect_robot()
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "execute_chess_move":
                    result = await self.execute_chess_move(
                        arguments["from_square"],
                        arguments["to_square"],
                        arguments.get("speed_factor", 1.0)
                    )
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "execute_single_trajectory":
                    result = await self.execute_single_trajectory(
                        arguments["square"],
                        arguments["action"],
                        arguments.get("speed_factor", 1.0)
                    )
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "list_available_trajectories":
                    result = await self.list_available_trajectories()
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "get_trajectory_info":
                    result = await self.get_trajectory_info(
                        arguments["square"],
                        arguments["action"]
                    )
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "arguments": arguments
                }, indent=2))]
    
    async def get_robot_status(self) -> Dict[str, Any]:
        """Get robot connection status"""
        return {
            "connected": self.robot is not None and self.robot.is_connected,
            "robot_type": "SO100" if self.robot else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def initialize_robot(self) -> Dict[str, Any]:
        """Initialize robot connection"""
        if self.robot and self.robot.is_connected:
            return {
                "status": "already_connected",
                "message": "Robot is already connected"
            }
        
        try:
            logger.info("Initializing SO100 robot...")
            
            # Redirect stdout to stderr during robot initialization
            with redirect_stdout(sys.stderr):
                config = So100RobotConfig()
                self.robot = ManipulatorRobot(config)
                self.robot.connect()
                self.executor = TrajectoryExecutor(self.robot)
            
            return {
                "status": "success",
                "message": "Robot connected successfully",
                "robot_type": "SO100"
            }
        except Exception as e:
            logger.error(f"Failed to initialize robot: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to initialize robot: {str(e)}"
            }
    
    async def disconnect_robot(self) -> Dict[str, Any]:
        """Disconnect from robot"""
        if not self.robot:
            return {
                "status": "not_connected",
                "message": "No robot connected"
            }
        
        try:
            # Redirect stdout to stderr during disconnection
            with redirect_stdout(sys.stderr):
                if self.robot.is_connected:
                    self.robot.disconnect()
                self.robot = None
                self.executor = None
            
            return {
                "status": "success",
                "message": "Robot disconnected successfully"
            }
        except Exception as e:
            logger.error(f"Error disconnecting robot: {str(e)}")
            return {
                "status": "error",
                "message": f"Error disconnecting: {str(e)}"
            }
    
    async def execute_chess_move(self, from_square: str, to_square: str, speed_factor: float) -> Dict[str, Any]:
        """Execute a complete chess move"""
        if not self.robot or not self.robot.is_connected:
            return {
                "status": "error",
                "message": "Robot not connected. Call initialize_robot first."
            }
        
        if not self.executor:
            return {
                "status": "error",
                "message": "Trajectory executor not initialized"
            }
        
        try:
            # Validate squares
            from_square = from_square.lower().strip()
            to_square = to_square.lower().strip()
            
            # Validate square format
            if not (len(from_square) == 2 and from_square[0] in 'abcdefgh' and from_square[1] in '12345678'):
                return {
                    "status": "error",
                    "message": f"Invalid from_square format: {from_square}"
                }
            
            if not (len(to_square) == 2 and to_square[0] in 'abcdefgh' and to_square[1] in '12345678'):
                return {
                    "status": "error",
                    "message": f"Invalid to_square format: {to_square}"
                }
            
            # Execute move in a separate thread to avoid blocking
            # Redirect stdout to stderr during execution
            loop = asyncio.get_event_loop()
            
            def execute_with_redirect():
                with redirect_stdout(sys.stderr):
                    self.executor.execute_move(from_square, to_square, speed_factor)
            
            await loop.run_in_executor(None, execute_with_redirect)
            
            # Disable robot motors after move to prevent overheating
            self.disable_robot_motors()
            
            return {
                "status": "success",
                "message": f"Move executed: {from_square} → {to_square}. Motors disabled to prevent overheating.",
                "from_square": from_square,
                "to_square": to_square,
                "speed_factor": speed_factor,
                "motors_disabled": True
            }
            
        except FileNotFoundError as e:
            return {
                "status": "error",
                "message": f"Trajectory not found: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error executing move: {str(e)}")
            return {
                "status": "error",
                "message": f"Error executing move: {str(e)}"
            }
    
    async def execute_single_trajectory(self, square: str, action: str, speed_factor: float) -> Dict[str, Any]:
        """Execute a single trajectory"""
        if not self.robot or not self.robot.is_connected:
            return {
                "status": "error",
                "message": "Robot not connected. Call initialize_robot first."
            }
        
        if not self.executor:
            return {
                "status": "error",
                "message": "Trajectory executor not initialized"
            }
        
        try:
            square = square.lower().strip()
            action = action.lower().strip()
            
            # Validate inputs
            if not (len(square) == 2 and square[0] in 'abcdefgh' and square[1] in '12345678'):
                return {
                    "status": "error",
                    "message": f"Invalid square format: {square}"
                }
            
            if action not in ['pickup', 'putdown']:
                return {
                    "status": "error",
                    "message": f"Invalid action: {action}. Must be 'pickup' or 'putdown'"
                }
            
            # Execute trajectory with stdout redirected
            loop = asyncio.get_event_loop()
            
            def execute_with_redirect():
                with redirect_stdout(sys.stderr):
                    self.executor.execute_trajectory(
                        square, action, speed_factor, False, 30.0
                    )
            
            await loop.run_in_executor(None, execute_with_redirect)
            
            # Note: We don't disable motors after single trajectory since
            # the user might want to execute another trajectory immediately
            
            return {
                "status": "success",
                "message": f"Trajectory executed: {square} {action}",
                "square": square,
                "action": action,
                "speed_factor": speed_factor
            }
            
        except FileNotFoundError as e:
            return {
                "status": "error",
                "message": f"Trajectory not found: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error executing trajectory: {str(e)}")
            return {
                "status": "error",
                "message": f"Error executing trajectory: {str(e)}"
            }
    
    async def list_available_trajectories(self) -> Dict[str, Any]:
        """List all available trajectories"""
        try:
            trajectories = self.manager.list_available_trajectories()
            
            # Organize by action type
            pickup_squares = sorted([s for s, a in trajectories if a == 'pickup'])
            putdown_squares = sorted([s for s, a in trajectories if a == 'putdown'])
            
            return {
                "status": "success",
                "total_trajectories": len(trajectories),
                "pickup_trajectories": pickup_squares,
                "putdown_trajectories": putdown_squares,
                "pickup_count": len(pickup_squares),
                "putdown_count": len(putdown_squares)
            }
            
        except Exception as e:
            logger.error(f"Error listing trajectories: {str(e)}")
            return {
                "status": "error",
                "message": f"Error listing trajectories: {str(e)}"
            }
    
    async def get_trajectory_info(self, square: str, action: str) -> Dict[str, Any]:
        """Get information about a specific trajectory"""
        try:
            square = square.lower().strip()
            action = action.lower().strip()
            
            # Load trajectory data
            filepath = self.manager.trajectories_dir / action / f"{square}.npz"
            if not filepath.exists():
                return {
                    "status": "error",
                    "message": f"Trajectory not found: {square} {action}"
                }
            
            data = dict(np.load(filepath))
            
            return {
                "status": "success",
                "square": square,
                "action": action,
                "duration": float(data['timestamps'][-1]),
                "frames": len(data['positions']),
                "file_size_kb": filepath.stat().st_size / 1024,
                "date_recorded": str(data.get('date', 'Unknown'))
            }
            
        except Exception as e:
            logger.error(f"Error getting trajectory info: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting trajectory info: {str(e)}"
            }
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, 
                write_stream, 
                self.server.create_initialization_options()
            )

async def main():
    """Main entry point"""
    server = ChessRobotMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 