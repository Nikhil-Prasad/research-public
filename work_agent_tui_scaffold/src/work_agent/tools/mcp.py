from __future__ import annotations

from work_agent.contracts import McpResult
from work_agent.policy import Policy, PolicyError


def call_mcp(*, server: str, tool: str, arguments: dict, policy: Policy) -> McpResult:
    if not policy.data.get("mcp", {}).get("enabled", False):
        raise PolicyError("MCP tool disabled")

    allowed = set(policy.data["mcp"].get("allowed_servers", []))
    if server not in allowed:
        raise PolicyError(f"MCP server not allowed: {server}")

    # Deliberately thin adapter.
    # Wire this to Python MCP SDK, local HTTP, stdio JSON-RPC, or direct Python functions.
    raise NotImplementedError("MCP adapter not wired yet. Implement server-specific call here.")
