#!/usr/bin/env python3
"""
Verify Chess Robot MCP Server Setup
"""

import sys
import json
import os
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_mark(success):
    return f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"

print("Chess Robot MCP Server Setup Verification")
print("=" * 50)

# Check Python environment
python_path = sys.executable
expected_python = "/home/charles/miniconda3/envs/so100/bin/python"
python_ok = python_path == expected_python

print(f"\n1. Python Environment:")
print(f"   {check_mark(python_ok)} Python path: {python_path}")
if not python_ok:
    print(f"   {YELLOW}   Expected: {expected_python}{RESET}")

# Check MCP installation
try:
    import mcp
    mcp_ok = True
    mcp_version = mcp.__version__ if hasattr(mcp, '__version__') else "Unknown"
except ImportError:
    mcp_ok = False
    mcp_version = "Not installed"

print(f"\n2. MCP Installation:")
print(f"   {check_mark(mcp_ok)} MCP package: {mcp_version}")

# Check robot dependencies
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    robot_deps_ok = True
except ImportError as e:
    robot_deps_ok = False
    robot_error = str(e)

print(f"\n3. Robot Dependencies:")
print(f"   {check_mark(robot_deps_ok)} LeRobot imports")
if not robot_deps_ok:
    print(f"   {RED}   Error: {robot_error}{RESET}")

# Check trajectory files
trajectories_dir = Path("chess_robot/trajectories")
trajectory_count = 0
if trajectories_dir.exists():
    for action in ['pickup', 'putdown']:
        action_dir = trajectories_dir / action
        if action_dir.exists():
            trajectory_count += len(list(action_dir.glob("*.npz")))

print(f"\n4. Trajectories:")
print(f"   {check_mark(trajectory_count > 0)} Found {trajectory_count} trajectories")

# Check Cursor configuration
config_path = Path("chess_robot/mcp_server/cursor_config.json")
config_ok = False
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    cmd = config.get("mcpServers", {}).get("chess-robot", {}).get("command", "")
    config_ok = cmd == expected_python

print(f"\n5. Cursor Configuration:")
print(f"   {check_mark(config_ok)} cursor_config.json uses correct Python")

# Overall status
all_ok = python_ok and mcp_ok and robot_deps_ok and trajectory_count > 0 and config_ok

print("\n" + "=" * 50)
if all_ok:
    print(f"{GREEN}✓ All checks passed! MCP server is ready to use.{RESET}")
    print("\nNext steps:")
    print("1. Copy configuration from cursor_config.json to Cursor Settings → MCP")
    print("2. Use Agent mode in Cursor chat")
    print("3. Start with: 'Initialize the chess robot'")
else:
    print(f"{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
    if not python_ok:
        print(f"\n{YELLOW}Tip: Run this script with:{RESET}")
        print(f"  {expected_python} {__file__}") 