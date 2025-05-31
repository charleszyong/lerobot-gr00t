#!/usr/bin/env python3
"""
Diagnose Cursor MCP connection issues
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def check_environment():
    """Check environment setup"""
    print("=== Environment Check ===")
    
    # Check Python path
    python_path = "/home/charles/miniconda3/envs/so100/bin/python"
    if os.path.exists(python_path):
        print(f"✓ Python found: {python_path}")
        # Check Python version
        result = subprocess.run([python_path, "--version"], capture_output=True, text=True)
        print(f"  Version: {result.stdout.strip()}")
    else:
        print(f"✗ Python not found: {python_path}")
    
    # Check server script
    server_path = "/home/charles/Projects/lerobot/chess_robot/mcp_server/run_server.py"
    if os.path.exists(server_path):
        print(f"✓ Server script found: {server_path}")
        # Check if executable
        if os.access(server_path, os.X_OK):
            print("  ✓ Script is executable")
        else:
            print("  ✗ Script is not executable")
    else:
        print(f"✗ Server script not found: {server_path}")
    
    # Check MCP package
    print("\n=== MCP Package Check ===")
    try:
        result = subprocess.run(
            [python_path, "-c", "import mcp; print(f'MCP version: {mcp.__version__}')"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "/home/charles/Projects/lerobot"}
        )
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print(f"✗ MCP package not found: {result.stderr}")
    except Exception as e:
        print(f"✗ Error checking MCP: {e}")
    
    # Check Cursor config
    print("\n=== Cursor Configuration ===")
    cursor_config = Path.home() / ".cursor" / "mcp.json"
    if cursor_config.exists():
        print(f"✓ Cursor config found: {cursor_config}")
        with open(cursor_config) as f:
            config = json.load(f)
            print(json.dumps(config, indent=2))
    else:
        print(f"✗ Cursor config not found: {cursor_config}")

def test_server_startup():
    """Test server startup"""
    print("\n=== Server Startup Test ===")
    
    cmd = [
        "/home/charles/miniconda3/envs/so100/bin/python",
        "/home/charles/Projects/lerobot/chess_robot/mcp_server/run_server.py"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/charles/Projects/lerobot"
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Wait a moment
        import time
        time.sleep(2)
        
        if proc.poll() is None:
            print("✓ Server started successfully")
            proc.terminate()
            proc.wait()
        else:
            _, stderr = proc.communicate()
            print(f"✗ Server failed to start")
            print(f"Exit code: {proc.returncode}")
            print(f"Error: {stderr}")
            
    except Exception as e:
        print(f"✗ Error starting server: {e}")

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n=== Suggested Actions ===")
    print("1. Restart Cursor completely (close all windows)")
    print("2. Check Cursor logs: View -> Toggle Developer Tools -> Console")
    print("3. Try removing and re-adding the MCP server in Cursor settings")
    print("4. Make sure the conda environment has all dependencies:")
    print("   conda activate so100")
    print("   pip install mcp")
    print("5. If still failing, try running Cursor from terminal to see errors:")
    print("   cursor")

if __name__ == "__main__":
    check_environment()
    test_server_startup()
    suggest_fixes() 