# Chess Robot MCP Server - Quick Start Guide

## 1. Prerequisites Check
```bash
# Activate conda environment
conda activate so100

# Check robot connection
ls /dev/ttyUSB*

# Test robot works
cd /home/charles/Projects/lerobot
python chess_robot/scripts/execute_trajectory.py
```

## 2. Install MCP
```bash
conda activate so100
pip install mcp
```

## 3. Test MCP Server
```bash
conda activate so100
python chess_robot/mcp_server/test_server.py
```

## 4. Configure Cursor

### Option A: Automatic Setup
```bash
./chess_robot/mcp_server/setup.sh
```

### Option B: Manual Setup
1. Open Cursor → Settings → MCP
2. Click "Add new global MCP server"
3. Add this configuration:
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
4. Save and refresh

## 5. Start Playing!

In Cursor chat (Agent mode):
```
You: Initialize the chess robot and let's play! I'll be white.
```

The AI will:
1. Connect to the robot
2. Confirm it's ready
3. Wait for your first move
4. Execute its moves on the physical board

## Common Commands

- **Start a game**: "Initialize the robot and let's play chess"
- **Make a move**: "I'll move my pawn from e2 to e4"
- **Check status**: "Is the robot still connected?"
- **End session**: "Disconnect the robot, game is over"

## Troubleshooting

**Robot not connecting?**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**MCP server not found?**
- Check the Python path in configuration matches: `/home/charles/miniconda3/envs/so100/bin/python`
- Ensure Python can find the lerobot module

**Missing trajectories?**
```bash
conda activate so100
python chess_robot/scripts/trajectory_utils.py progress
```

Enjoy playing chess with your AI-powered robot! 