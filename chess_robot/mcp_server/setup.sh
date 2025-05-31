#!/bin/bash
# Setup script for Chess Robot MCP Server

echo "Setting up Chess Robot MCP Server..."
echo "===================================="

# Get the absolute path of the lerobot project
LEROBOT_PATH=$(cd "$(dirname "$0")/../.." && pwd)
echo "LeRobot path: $LEROBOT_PATH"

# Set the conda environment Python path
CONDA_PYTHON="/home/charles/miniconda3/envs/so100/bin/python"
echo "Using Python: $CONDA_PYTHON"

# Install MCP if not already installed
echo -e "\nInstalling MCP in conda environment..."
$CONDA_PYTHON -m pip install mcp

# Create Cursor configuration
CURSOR_CONFIG_DIR="$HOME/.cursor"
mkdir -p "$CURSOR_CONFIG_DIR"

# Generate the MCP configuration
echo -e "\nGenerating Cursor MCP configuration..."
cat > "$CURSOR_CONFIG_DIR/mcp_chess_robot.json" <<EOF
{
  "mcpServers": {
    "chess-robot": {
      "command": "$CONDA_PYTHON",
      "args": ["$LEROBOT_PATH/chess_robot/mcp_server/run_server.py"],
      "env": {
        "PYTHONPATH": "$LEROBOT_PATH"
      }
    }
  }
}
EOF

echo -e "\nSetup complete!"
echo -e "\nTo use the MCP server in Cursor:"
echo "1. Open Cursor Settings → MCP"
echo "2. Click 'Add new global MCP server'"
echo "3. Copy the content from: $CURSOR_CONFIG_DIR/mcp_chess_robot.json"
echo "4. Save and refresh the MCP servers"
echo -e "\nAlternatively, you can manually add the configuration shown above." 