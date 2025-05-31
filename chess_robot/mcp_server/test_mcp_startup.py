#!/usr/bin/env python3
"""
Test MCP server startup
"""
import subprocess
import json
import time

def test_mcp_startup():
    """Test if the MCP server starts correctly"""
    # Start the server process
    cmd = [
        "/home/charles/miniconda3/envs/so100/bin/python",
        "/home/charles/Projects/lerobot/chess_robot/mcp_server/run_server.py"
    ]
    env = {
        "PYTHONPATH": "/home/charles/Projects/lerobot",
        "PATH": "/home/charles/miniconda3/envs/so100/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    }
    
    try:
        # Start the process
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "roots": {},
                    "experimental": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        proc.stdin.write(json.dumps(initialize_request) + "\n")
        proc.stdin.flush()
        
        # Give it a moment to respond
        time.sleep(1)
        
        # Check if process is still running
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            print(f"Server exited with code: {proc.returncode}")
            print(f"Stderr: {stderr}")
            return False
        
        print("Server started successfully!")
        
        # Clean up
        proc.terminate()
        proc.wait()
        return True
        
    except Exception as e:
        print(f"Error testing server: {e}")
        return False

if __name__ == "__main__":
    test_mcp_startup() 