#!/usr/bin/env python3
"""
Fix Cursor MCP Configuration
This script ensures the MCP server configuration is in all possible locations
"""

import json
import os
from pathlib import Path
import shutil

# MCP Configuration
MCP_CONFIG = {
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

# Possible locations for MCP config
config_locations = [
    Path.home() / ".cursor" / "mcp.json",
    Path.home() / ".config" / "Cursor" / "User" / "mcp.json",
    Path.home() / ".config" / "cursor" / "mcp.json",
]

# Ensure all directories exist and write config
for config_path in config_locations:
    print(f"\nChecking: {config_path}")
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the configuration
    try:
        with open(config_path, 'w') as f:
            json.dump(MCP_CONFIG, f, indent=2)
        print(f"✓ Written MCP config to: {config_path}")
    except Exception as e:
        print(f"✗ Failed to write to {config_path}: {e}")

# Also update Cursor's main settings.json to include MCP config
settings_path = Path.home() / ".config" / "Cursor" / "User" / "settings.json"
if settings_path.exists():
    print(f"\nUpdating Cursor settings at: {settings_path}")
    try:
        # Read existing settings
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Add MCP configuration
        settings.update(MCP_CONFIG)
        
        # Write back
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print("✓ Updated Cursor settings.json with MCP config")
    except Exception as e:
        print(f"✗ Failed to update settings.json: {e}")

print("\n" + "="*50)
print("MCP Configuration Setup Complete!")
print("="*50)
print("\nNext steps:")
print("1. Completely quit Cursor (Cmd/Ctrl+Q)")
print("2. Start Cursor again")
print("3. Open a new chat in Agent mode")
print("4. The chess-robot tools should now be available")
print("\nIf still not working, try:")
print("- Check View → Toggle Developer Tools → Console for errors")
print("- Make sure you're using Agent mode (not regular chat)")
print("- Try: 'What MCP tools do you have available?'") 