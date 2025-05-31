# Using Claude with MCP for Robot Control

This guide explains how to set up and use the MCP (Model-Connector-Planner) architecture with Claude 3.7 Sonnet to control your Tic-Tac-Toe playing robot.

## Overview

The MCP implementation consists of two main components:

1. **MCP Server** (`tictac_mcp_server.py`): A Flask server that provides API endpoints to interact with the robot
2. **Claude MCP Client** (`tictac_claude_mcp.py`): A client that uses Claude to analyze the game board and make decisions

This approach allows Claude to interact with the robot through structured API calls rather than relying on traditional prompt engineering.

## Setup Instructions

### Step 1: Set Up the Environment

Ensure you have all the required dependencies:

```bash
pip install flask flask-cors requests numpy opencv-python torch anthropic
```

### Step 2: Get an Anthropic API Key

1. Visit [Anthropic's Console](https://console.anthropic.com/)
2. Sign up for an account or log in to your existing account
3. Navigate to the API Keys section in your account settings
4. Create a new API key (keep this secure - don't share it or commit it to version control)

### Step 3: Set Up the MCP Server

1. Start the GR00T VLA server according to the usual instructions
2. In a separate terminal, start the MCP server:

```bash
# Start the MCP server
python tictac_mcp_server.py --vla-host localhost --vla-port 5555 --cam-idx 8
```

The MCP server will start on port 8000 by default. You can verify it's working by visiting `http://localhost:8000` in your browser.

### Step 4: Run the Claude MCP Client

In another terminal, set your Anthropic API key as an environment variable:

```bash
# Unix/Linux/macOS
export ANTHROPIC_API_KEY=your_api_key_here

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

Then run the Claude MCP client:

```bash
python tictac_claude_mcp.py --server-url http://localhost:8000 --moves 5
```

## How It Works

### MCP Server

The MCP server provides several API endpoints:

1. `GET /api/get_board_state`: Returns the current board image
2. `GET /api/get_available_positions`: Returns all valid positions on the board
3. `POST /api/execute_move`: Executes a move at the specified position
4. `POST /api/reset`: Resets the robot to home position
5. `GET /api/info`: Returns information about the server

These endpoints allow Claude to:
- Observe the current state of the game
- Know what moves are available
- Send commands to the robot

### Claude MCP Client

The Claude MCP client:

1. Captures the current board state through the MCP server
2. Sends the image to Claude along with available positions
3. Asks Claude to analyze the board and select the best move
4. Extracts the chosen position from Claude's response
5. Sends the position back to the MCP server to execute the move

## Command Line Options

### MCP Server

```
python tictac_mcp_server.py [options]

Options:
  --host HOST         Server host (default: 0.0.0.0)
  --port PORT         Server port (default: 8000)
  --vla-host VLA_HOST VLA server host (default: localhost)
  --vla-port VLA_PORT VLA server port (default: 5555)
  --cam-idx CAM_IDX   Camera index (default: 8)
```

### Claude MCP Client

```
python tictac_claude_mcp.py [options]

Options:
  --server-url SERVER_URL  MCP server URL (default: http://localhost:8000)
  --api-key API_KEY        Claude API key (default: use ANTHROPIC_API_KEY from environment)
  --moves MOVES            Number of moves to play (default: 5)
  --model MODEL            Claude model to use (default: claude-3-7-sonnet-20250219)
```

## Troubleshooting

### MCP Server Issues

- **Robot not moving**: Check if the VLA server is running and the correct host/port are specified
- **Camera not working**: Verify the camera index is correct
- **Server not starting**: Check for port conflicts or missing dependencies

### Claude MCP Client Issues

- **API key errors**: Ensure your Anthropic API key is set correctly
- **Connection errors**: Make sure the MCP server is running and the URL is correct
- **Claude not making good moves**: You can adjust the system prompt in the client code

## Extending the System

You can extend this MCP system in several ways:

1. **Additional robot capabilities**: Add new API endpoints to the MCP server
2. **Better vision**: Enhance the image processing to provide Claude with more information
3. **Advanced strategies**: Modify the system prompt to teach Claude better game strategies
4. **Multi-turn planning**: Add endpoints for planning sequences of moves

## Example Workflow

1. Start VLA server: `python scripts/inference_service.py --server ...`
2. Start MCP server: `python tictac_mcp_server.py`
3. Start Claude client: `python tictac_claude_mcp.py`
4. Follow the prompts to play the game
5. Make your moves when prompted
6. Watch Claude analyze the board and control the robot to make its moves

## Web Interface

The MCP server also provides a simple web interface at `http://localhost:8000` where you can:
- See the current board state
- Manually refresh the image
- Reset the robot

This interface is useful for debugging and verifying that the MCP server is working correctly.