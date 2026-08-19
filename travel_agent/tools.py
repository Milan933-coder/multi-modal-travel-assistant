"""The manual tool execution engine — this is Distinction 1.
We deliberately skip ToolNode and wire up the raw tool-calling protocol ourselves."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from langchain_core.messages import ToolMessage

from .data import mock_image_search, mock_web_search, mock_weather_forecast
from .knowledge import LocalKnowledgeStore
from .models import CityKnowledge, ToolCall, WeatherPoint


# These mirror what a real LLM would see as available functions.
# The schema isn't consumed at runtime here — it's for documentation
# and to show we understand the tool-calling contract.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_local_knowledge",
            "description": "Lookup verified facts about an indexed city (Paris, Tokyo, New York) in the local ChromaDB vector store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The destination city name (e.g. Paris, Tokyo, New York)"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mock_web_search",
            "description": "Perform web search to discover travel facts for destinations outside the internal vector catalog (e.g. Kyoto, Snohomish, Rome).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The destination city name to search"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_weather_forecast",
            "description": "Retrieve a 7-day weather forecast, daily temperatures, and precipitation probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Target city"},
                    "days": {"type": "integer", "description": "Number of forecast days (1-7)", "default": 7},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_location_images",
            "description": "Retrieve verified high-resolution photography URLs for the destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Target city name"}
                },
                "required": ["city"],
            },
        },
    },
]


def manual_execute_tool_calls(
    raw_tool_calls: list[ToolCall | dict[str, Any]],
    knowledge_store: LocalKnowledgeStore,
) -> tuple[dict[str, Any], list[ToolMessage], list[dict[str, Any]], list[str]]:
    """The heart of Distinction 1 — we do what ToolNode does, but by hand.

    Walk through each raw tool call, look it up in our registry, run it,
    and package the result as a proper ToolMessage. Tracks latency for
    every call so we can show it in the trace tab.
    """
    updates: dict[str, Any] = {}
    messages: list[ToolMessage] = []
    trace: list[dict[str, Any]] = []
    errors: list[str] = []

    # Map tool names to actual callable functions
    registry: dict[str, Callable[..., Any]] = {
        "lookup_local_knowledge": knowledge_store.get_city,
        "mock_web_search": mock_web_search,
        "fetch_weather_forecast": mock_weather_forecast,
        "fetch_location_images": mock_image_search,
    }

    if not isinstance(raw_tool_calls, list):
        raw_tool_calls = []

    for index, raw_call in enumerate(raw_tool_calls):
        start_time = time.perf_counter()

        # Guard against garbage — if someone passes None or a string, catch it
        if not isinstance(raw_call, dict):
            call_id = f"malformed-call-{index}"
            err_msg = "Malformed tool call: payload must be a JSON object with id, name, and args."
            trace.append({
                "id": call_id,
                "name": "unknown",
                "status": "failed",
                "error": err_msg,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            })
            errors.append(err_msg)
            messages.append(ToolMessage(content=json.dumps({"error": err_msg}), tool_call_id=call_id, name="unknown"))
            continue

        call_id = str(raw_call.get("id") or f"tool-call-{index}")
        name = str(raw_call.get("name") or "")
        raw_args = raw_call.get("args", {})

        # Some LLMs emit args as a JSON string instead of a dict — handle both
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"city": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        try:
            if name not in registry:
                raise ValueError(f"Tool '{name}' is not registered in the agent's tool catalog.")

            tool_func = registry[name]
            result = tool_func(**args)

            if result is None:
                raise ValueError(f"No destination knowledge found for '{args.get('city', 'the requested city')}'.")

            # Serialize and stash the result in the right state key
            if isinstance(result, CityKnowledge):
                serialized = result.model_dump_json()
                updates["knowledge"] = result.model_dump(mode="json")
            elif isinstance(result, list) and result and isinstance(result[0], WeatherPoint):
                serialized = json.dumps([p.model_dump(mode="json") for p in result])
                updates["weather_forecast"] = [p.model_dump(mode="json") for p in result]
            elif isinstance(result, list):
                serialized = json.dumps(result)
                if name == "fetch_location_images":
                    updates["image_urls"] = result
            else:
                serialized = json.dumps(result, default=str)
                if name == "mock_web_search":
                    updates["web_context"] = serialized

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            messages.append(ToolMessage(content=serialized, tool_call_id=call_id, name=name))
            trace.append({
                "id": call_id,
                "name": name,
                "args": args,
                "status": "completed",
                "latency_ms": elapsed_ms,
            })

        except Exception as exc:
            # Don't let one bad tool call nuke the whole pipeline
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"Tool '{name}' execution failed: {exc}"
            errors.append(err_msg)
            trace.append({
                "id": call_id,
                "name": name,
                "args": args,
                "status": "failed",
                "error": str(exc),
                "latency_ms": elapsed_ms,
            })
            messages.append(
                ToolMessage(
                    content=json.dumps({"error": str(exc)}),
                    tool_call_id=call_id,
                    name=name or "unknown",
                )
            )

    return updates, messages, trace, errors
