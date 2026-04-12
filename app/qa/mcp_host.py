"""
Async MCP host helpers for the Q/A runtime.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import AnyUrl, TypeAdapter

_URL = TypeAdapter(AnyUrl)
_ROOT = Path(__file__).resolve().parents[2]


def _extract_text_content(call_result) -> str:
    texts = []
    for item in call_result.content:
        if getattr(item, "type", "") == "text":
            texts.append(item.text)
    return "\n".join(texts).strip()


def _decode_json_like(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def decode_tool_result(call_result) -> Any:
    structured = call_result.structuredContent
    if structured is not None:
        # FastMCP may wrap string tool returns as {"result": "<json-string>"}.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            decoded = _decode_json_like(structured["result"])
            if decoded != structured["result"]:
                return decoded
        return structured
    text = _extract_text_content(call_result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def decode_resource_result(read_result) -> Any:
    if not read_result.contents:
        return {}
    content = read_result.contents[0]
    text = getattr(content, "text", "")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


@asynccontextmanager
async def qa_mcp_session():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.qa.mcp_server"],
        cwd=_ROOT,
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def read_json_resource(session: ClientSession, uri: str) -> Any:
    result = await session.read_resource(_URL.validate_python(uri))
    return decode_resource_result(result)
