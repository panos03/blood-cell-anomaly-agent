"""Proves the MCP server actually works end-to-end, independent of any LLM.

Spawns agent/mcp_server.py as a real subprocess (the same way Claude Desktop
or any other MCP client would) and talks to it over the MCP stdio protocol:
discovers its tools, then calls each one and checks the result against
agent/tools.py called directly. No Claude API call, no cost - this only
tests the server/protocol plumbing, not the agent's reasoning.

Usage:
    python -m agent.mcp_smoke_test
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent import tools as t


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"])


async def main() -> None:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(tl.name for tl in tools.tools)
            expected = sorted(["get_flagged_cells", "get_cluster_summary", "compare_to_baseline"])
            assert names == expected, f"Tool list mismatch: {names} != {expected}"
            print(f"OK  server exposes the expected tools: {names}")

            checks = [
                ("get_flagged_cells", {"holdout_class": "erythroblast"}, t.get_flagged_cells("erythroblast")),
                ("get_cluster_summary", {"k": 2}, t.get_cluster_summary(k=2)),
                (
                    "compare_to_baseline",
                    {"holdout_class": "monocyte", "metric": "recall"},
                    t.compare_to_baseline("monocyte", "recall"),
                ),
            ]

            for name, args, expected_result in checks:
                result = await session.call_tool(name, args)
                assert not result.is_error, f"{name}({args}) returned an error: {result.content}"
                actual = json.loads(result.content[0].text)
                assert actual == expected_result, (
                    f"{name}({args}) over MCP != direct call:\n  MCP:    {actual}\n  direct: {expected_result}"
                )
                print(f"OK  {name}({args}) over MCP matches calling tools.py directly")

    print("\nAll MCP smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
