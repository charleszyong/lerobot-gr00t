#!/usr/bin/env python3
"""
Minimal MCP test server to verify Cursor connection
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

async def main():
    # Create server with proper name
    server = Server("test-server")
    
    # Register tool handler
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="test_ping",
                description="Simple test tool that returns pong",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    # Register call handler
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "test_ping":
            return [TextContent(type="text", text="pong! MCP server is working!")]
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 