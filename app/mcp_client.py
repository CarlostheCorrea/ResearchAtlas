"""
Lightweight MCP client used by agents to call MCP tools.
Phase 3 — Week 8: MCP Foundations. Agents call MCP tools through this client —
they never import tool implementations directly.
"""
import httpx
from app.config import MCP_SERVER_PORT

MCP_BASE_URL = f"http://localhost:{MCP_SERVER_PORT}"


class MCPClient:
    """HTTP client for the MCP tool server."""

    def __init__(self, base_url: str = MCP_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout

    def call_tool(self, tool: str, params: dict) -> dict:
        """Call a named MCP tool with the given parameters."""
        try:
            resp = httpx.post(
                f"{self.base_url}/call",
                json={"tool": tool, "params": params},
                timeout=self.timeout,
            )
            if not resp.is_success:
                # Log the actual error detail from the server response body
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                print(f"[mcp_client] call_tool({tool}) HTTP {resp.status_code}: {detail}")
                return {"error": detail}
            return resp.json()
        except Exception as e:
            print(f"[mcp_client] call_tool({tool}) failed: {e}")
            return {"error": str(e)}


# Module-level singleton — one client per process
_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client
