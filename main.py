import json
import asyncio
from fastmcp import Client

async def main():
    async with Client("server.py") as client:
        print(f"Client connected: {client.is_connected()}")

        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        result = await client.call_tool("DemoCustomWidget", {"environ": {}, "prompt": "World"})
        # The result should match the OutputsSchema defined in demo.py with 'reply' field
        result = json.loads(result[0].text)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())