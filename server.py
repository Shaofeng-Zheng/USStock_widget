import argparse
from fastmcp import FastMCP

parser = argparse.ArgumentParser(description='Demo Custom Widget MCP Server')
parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server to')
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--transport', default="stdio", help='Transport protocol for the main proxy server (stdio or streamable-http)')
args = parser.parse_args()

mcp = FastMCP("US Stock")

@mcp.tool(name="US Stock")
def demo_tool(environ: dict, prompt: str) -> str:
    return {
        "reply": "Hello world! Welcome to try custom widgets"
    }

if __name__ == "__main__":
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)