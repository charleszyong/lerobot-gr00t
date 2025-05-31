# Chess Robot MCP Server

This MCP (Model Context Protocol) server allows an AI agent in Cursor to control a physical chess robot, enabling it to play chess with you by moving real pieces on a physical board.

## Overview

The Chess Robot MCP Server provides the following capabilities:
- **Robot Control**: Initialize, connect, and disconnect the SO100 robot
- **Chess Moves**: Execute complete chess moves (pickup from one square, put down on another)
- **Individual Trajectories**: Execute single pickup or putdown actions
- **Trajectory Management**: List available trajectories and get information about them

## Prerequisites

1. **SO100 Robot**: The physical robot must be connected to your computer
2. **Recorded Trajectories**: You must have pre-recorded trajectories for each chess square
3. **Python Environment**: Python 3.8 or higher with required dependencies (conda env: so100)
4. **Cursor IDE**: Latest version with MCP support

## Installation

### 1. Install Dependencies

First, ensure you have the main lerobot dependencies installed in the conda environment:
```bash
cd /home/charles/Projects/lerobot
conda activate so100
pip install -e .
```

Then install MCP-specific dependencies:
```bash
conda activate so100
pip install mcp
```

### 2. Verify Robot Connection

Before using the MCP server, verify your robot works:
```bash
conda activate so100
python chess_robot/scripts/execute_trajectory.py
```

### 3. Check Available Trajectories

List recorded trajectories:
```bash
conda activate so100
python chess_robot/scripts/trajectory_utils.py progress
```

## Setting Up MCP Server in Cursor

### Method 1: Global Configuration

1. Open Cursor Settings
2. Navigate to the MCP section
3. Click "Add new global MCP server"
4. Add this configuration to the JSON file:

```json
{
  "mcpServers": {
    "chess-robot": {
      "command": "/home/charles/miniconda3/envs/so100/bin/python",
      "args": ["/home/charles/Projects/lerobot/chess_robot/mcp_server/run_server.py"],
      "env": {
        "PYTHONPATH": "/home/charles/Projects/lerobot"
      }
    }
  }
}
```

5. Save the configuration
6. Click the refresh icon to reload MCP servers
7. Verify the server shows a green dot (running status)

### Method 2: Project-Specific Configuration

Create a `.cursor/mcp.json` file in your project root with the same configuration as above.

## Using the Chess Robot in Cursor

### 1. Switch to Agent Mode

In Cursor's chat panel:
- Open the chat (Cmd+I on Mac, Ctrl+I on Windows/Linux)
- Select "Cursor Agent" mode (not "Ask" or "Manual")

### 2. Available Commands

The AI agent can use these tools:

#### Initialize Robot
```
"Initialize the chess robot"
```

#### Execute Chess Move
```
"Move the piece from e2 to e4"
"Execute chess move from d7 to d5"
"Play knight from g1 to f3"
```

#### List Available Trajectories
```
"What chess squares have recorded trajectories?"
"Show me all available positions"
```

#### Get Trajectory Info
```
"Get information about the e4 pickup trajectory"
"What's the duration of the d5 putdown trajectory?"
```

#### Disconnect Robot
```
"Disconnect the robot"
```

### 3. Example Chess Game Session

Here's an example conversation with the AI agent:

```
You: Initialize the chess robot and let's play a game. I'll play white.

Agent: I'll initialize the chess robot for our game.
[Uses initialize_robot tool]

You: I'll start with e2 to e4

Agent: I'll move my pawn from e7 to e5 in response.
[Uses execute_chess_move tool with from_square="e7", to_square="e5"]

You: Knight from g1 to f3

Agent: I'll develop my knight from b8 to c6.
[Uses execute_chess_move tool with from_square="b8", to_square="c6"]
```

## MCP Server Tools Reference

### get_robot_status
Check if the robot is connected and ready.
- **Input**: None
- **Output**: Connection status and robot type

### initialize_robot
Initialize connection to the chess robot.
- **Input**: None
- **Output**: Success/error status

### execute_chess_move
Execute a complete chess move.
- **Input**: 
  - `from_square` (string): Source square (e.g., "e2")
  - `to_square` (string): Destination square (e.g., "e4")
  - `speed_factor` (number, optional): Speed multiplier (0.1-2.0, default 1.0)
- **Output**: Success/error status with move details

### execute_single_trajectory
Execute a single pickup or putdown action.
- **Input**:
  - `square` (string): Chess square (e.g., "e4")
  - `action` (string): "pickup" or "putdown"
  - `speed_factor` (number, optional): Speed multiplier (0.1-2.0, default 1.0)
- **Output**: Success/error status with trajectory details

### list_available_trajectories
List all recorded trajectories.
- **Input**: None
- **Output**: Lists of pickup and putdown trajectories by square

### get_trajectory_info
Get information about a specific trajectory.
- **Input**:
  - `square` (string): Chess square (e.g., "e4")
  - `action` (string): "pickup" or "putdown"
- **Output**: Duration, frame count, file size, and recording date

### disconnect_robot
Disconnect from the chess robot.
- **Input**: None
- **Output**: Success/error status

## Troubleshooting

### MCP Server Not Starting
1. Check Python path in configuration (should be `/home/charles/miniconda3/envs/so100/bin/python`)
2. Verify all dependencies are installed in the conda environment
3. Check server logs in Cursor's MCP settings

### Robot Connection Issues
1. Ensure robot is powered on and connected via USB
2. Check robot permissions: `sudo chmod 666 /dev/ttyUSB0`
3. Verify robot works with direct scripts before using MCP

### Missing Trajectories
1. Record missing trajectories using:
   ```bash
   conda activate so100
   python chess_robot/scripts/record_chess_trajectories.py
   ```
2. Verify trajectories with:
   ```bash
   conda activate so100
   python chess_robot/scripts/trajectory_utils.py verify
   ```

### Performance Issues
- Adjust `speed_factor` parameter (0.5 for slower, 2.0 for faster)
- Check robot motor temperature if movements are sluggish

## Advanced Usage

### Custom Game Rules

You can set custom rules for the AI agent in Cursor Settings → Rules:

```
When playing chess with the robot:
1. Always confirm moves before executing
2. Use speed_factor 0.8 for careful moves
3. Announce each move clearly
4. Wait for human confirmation after each robot move
```

### Integration with Chess Engines

The MCP server can be extended to integrate with chess engines like Stockfish for AI gameplay. The current implementation focuses on move execution, but game logic can be added.

## Safety Considerations

1. **Robot Safety**: Always supervise the robot during operation
2. **Collision Avoidance**: Ensure the chess board is clear before moves
3. **Emergency Stop**: Press ESC during trajectory execution to abort
4. **Power Management**: Disconnect robot when not in use to prevent overheating

## Contributing

To extend the MCP server:

1. Add new tools in `server.py`
2. Implement tool handlers with proper error handling
3. Update the tool list in `list_tools()`
4. Test thoroughly with the physical robot

## License

This MCP server is part of the lerobot project and follows its licensing terms. 