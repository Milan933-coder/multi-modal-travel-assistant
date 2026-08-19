"""The LangGraph pipeline — nodes, edges, routing logic, and the LLM synthesis step.
This is where all the magic happens: parse → route → fetch → finalize."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .data import (
    CITY_FACTS,
    display_city,
    mock_image_search,
    mock_weather_forecast,
    mock_web_search,
    normalize_city,
)
from .knowledge import LocalKnowledgeStore
from .models import (
    BudgetEstimate,
    CityKnowledge,
    FamousDish,
    LocalCulture,
    LocalEvent,
    Neighborhood,
    TravelResponse,
    TravelState,
    WeatherPoint,
)
from .tools import manual_execute_tool_calls

load_dotenv()

MAX_QUERY_CHARS = 500

# Common words that look like city names but aren't — prevents
# "tell" or "weather" from being parsed as a destination
STOP_WORDS = {
    "next week", "this week", "next month", "tomorrow", "today", "yesterday",
    "what", "how", "when", "where", "why", "who", "tell", "show", "weather",
    "forecast", "temperature", "rain", "sunny", "guide", "photos", "pictures",
    "about", "explore", "visit", "trip", "travel", "around"
}


def _extract_city(query: str, previous_city: str | None) -> str:
    """Figure out what city the user is asking about.

    Priority order:
    1. Exact match against our known ChromaDB cities (+ aliases like 'NYC')
    2. Regex extraction from natural phrasing like 'Tell me about Kyoto'
    3. Fall back to the previous city from conversation memory
    4. Last resort: grab the last meaningful word(s) from the query
    """
    query = query[:MAX_QUERY_CHARS].strip()
    lowered = query.lower()

    # Check against our vector store cities first — these get priority routing
    for known_key, fact in CITY_FACTS.items():
        candidates = [known_key, fact.city.lower()]
        if known_key == "new york":
            candidates.extend(["nyc", "new york city", "the big apple", "manhattan"])
        if any(re.search(rf"\b{re.escape(c)}\b", lowered) for c in candidates):
            return fact.city

    # Try common travel query patterns: "about X", "visit X", "trip to X", etc.
    patterns = [
        r"\b(?:about|in|to|for|of|visit|explore|travel to|guide for|trip to)\s+([A-Za-z\s'-]{2,40}?)(?:\?|$|\s+(?:weather|forecast|photos?|images?|travel|guide|today|tomorrow|next week|next month))",
        r"^([A-Za-z\s'-]{2,30})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = re.sub(r"[\r\n\t]+", " ", match.group(1)).strip(" .,!?'\"")
            candidate_clean = candidate.lower().strip()
            if candidate_clean and candidate_clean not in STOP_WORDS and len(candidate) >= 2:
                return display_city(candidate)

    # If they said "what about next week?" we already know the city from last turn
    if previous_city:
        return previous_city

    # Hail mary — just grab whatever non-stop-word is left
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", query) if w.lower() not in STOP_WORDS]
    if words:
        return display_city(" ".join(words[-2:]) if len(words) > 1 else words[0])
    return "Unknown destination"


def _is_weather_follow_up(query: str, previous_city: str | None) -> bool:
    """Detect if this is a 'what about next week?' style follow-up (Distinction 3).

    Returns True only when:
    - There's a previous city in memory
    - The query mentions weather/time words
    - The query does NOT mention a new city or ask for a full overview
    """
    if not previous_city:
        return False
    lowered = query.lower()
    weather_terms = (
        "weather", "forecast", "temperature", "temp", "rain", "rainy", "sunny",
        "climate", "next week", "tomorrow", "next few days", "next month",
        "cold", "hot", "warm", "humid", "weekend", "is it raining"
    )
    explicit_overview_terms = (
        "tell me about", "guide to", "explore", "visit", "what to do",
        "highlights of", "photos of", "images of", "food in", "dishes in"
    )

    # If they mention a different city, it's a new query — not a follow-up
    for known_key, fact in CITY_FACTS.items():
        if fact.city.lower() != previous_city.lower():
            if re.search(rf"\b{re.escape(known_key)}\b", lowered) or re.search(rf"\b{re.escape(fact.city.lower())}\b", lowered):
                return False

    has_weather = any(term in lowered for term in weather_terms)
    has_explicit_overview = any(term in lowered for term in explicit_overview_terms)
    return has_weather and not has_explicit_overview


def _generate_weather_packing_list(weather_points: list[WeatherPoint]) -> list[str]:
    """Build a packing checklist that actually makes sense for the forecast.
    Rainy week? Pack an umbrella. Freezing? Bring layers. Hot? Sunscreen."""
    if not weather_points:
        return [
            "Comfortable broken-in walking shoes (15k+ steps/day)",
            "Universal travel power adapter & USB-C portable charger",
            "Reusable insulated water bottle",
            "Daypack with anti-theft zipper compartments",
        ]

    max_rain = max(p.precipitation_probability for p in weather_points)
    min_temp = min(p.temperature_c for p in weather_points)
    max_temp = max(p.temperature_c for p in weather_points)
    max_wind = max(p.wind_kmh for p in weather_points)

    checklist: list[str] = [
        "Comfortable broken-in walking shoes (15k+ steps/day)",
        "Universal travel power adapter & high-capacity battery bank",
    ]

    if max_rain >= 35:
        checklist.append(f"Compact windproof umbrella & water-resistant shell jacket ({max_rain}% rain expected)")
    if min_temp < 15:
        checklist.append(f"Layerable merino fleece, light sweater & evening scarf (lows near {min_temp:.0f}°C)")
    if max_temp >= 24:
        checklist.append(f"Breathable linen tops, UV sunglasses & SPF 50+ sunscreen (highs near {max_temp:.0f}°C)")
    if max_wind >= 20:
        checklist.append(f"Windbreaker jacket & secure hat clip (wind gusts up to {max_wind} km/h)")

    checklist.append("Crossbody bag or lightweight daypack with secure zip closures")
    checklist.append("Digital copies of passport, hotel confirmations & local transit cards")
    return checklist


# ---------------------------------------------------------------------------
#  GRAPH NODES — each function is one step in the LangGraph pipeline
# ---------------------------------------------------------------------------

def _parse_request(state: TravelState) -> dict[str, Any]:
    """First node: figure out what the user wants and set up the state for routing."""
    raw_query = state.get("user_query", "")
    query = raw_query.strip() if isinstance(raw_query, str) else str(raw_query or "").strip()
    query = query[:MAX_QUERY_CHARS]

    target_lang = state.get("target_language", "English") or "English"
    previous_city = state.get("city") if isinstance(state.get("city"), str) else None
    existing_errors = state.get("errors") or []
    refresh_weather_only = _is_weather_follow_up(query, previous_city)
    city = previous_city if refresh_weather_only else (_extract_city(query, previous_city) if query else "Unknown destination")

    updates: dict[str, Any] = {
        "user_query": query,
        "target_language": target_lang,
        "city": city,
        "request_kind": "weather_follow_up" if refresh_weather_only else "city_overview",
        "empty_request": not bool(query),
        "refresh_weather_only": refresh_weather_only,
        "errors": [],
        "pending_tool_calls": [],
        "messages": [HumanMessage(content=query)],
        "route_reason": "",
        "error_start_index": len(existing_errors),
        "start_time": time.perf_counter(),
    }

    # On a fresh city query, wipe previous results so we don't show stale data
    if not refresh_weather_only:
        updates.update({
            "knowledge": None,
            "web_context": "",
            "weather_forecast": [],
            "image_urls": [],
            "packing_essentials": [],
            "result": None,
        })
    return updates


def _route_after_parse(
    state: TravelState,
    knowledge_store: LocalKnowledgeStore | None = None,
) -> Literal["empty", "reuse_context", "local_tool", "web_tool"]:
    """The conditional edge — decides which path the graph takes.

    Four possible routes:
    - 'empty'          → user sent blank input
    - 'reuse_context'  → follow-up query, city already in memory (Distinction 3)
    - 'local_tool'     → city exists in ChromaDB, use vector store
    - 'web_tool'       → unknown city, fire up web search
    """
    if state.get("empty_request"):
        return "empty"
    if state.get("refresh_weather_only") and (state.get("knowledge") or state.get("web_context")):
        return "reuse_context"

    store = knowledge_store or LocalKnowledgeStore()
    city = state.get("city", "")
    if store.has_city(city):
        return "local_tool"
    return "web_tool"


def _empty_request(state: TravelState) -> dict[str, Any]:
    """Handle blank queries — just tell the user to type something."""
    return {
        "route_reason": "No destination was provided; external tools and inference were skipped.",
        "errors": ["Please enter a valid city or destination to start an exploration."],
    }


def _prepare_local_tool(state: TravelState) -> dict[str, Any]:
    """Set up a raw tool call to hit ChromaDB — the graph's manual_tool_executor will run it."""
    city = state.get("city", "Unknown destination")
    return {
        "route_reason": f"'{city}' is indexed in the pre-populated ChromaDB vector catalog; routed to internal knowledge store.",
        "pending_tool_calls": [
            {
                "id": f"call_chroma_{int(time.time()*1000)}",
                "name": "lookup_local_knowledge",
                "args": {"city": city},
            }
        ],
    }


def _prepare_web_tool(state: TravelState) -> dict[str, Any]:
    """Set up a raw tool call for web search — city isn't in our local store."""
    city = state.get("city", "Unknown destination")
    return {
        "route_reason": f"'{city}' is outside the internal vector catalog; dynamically switched to Web Search routing path.",
        "pending_tool_calls": [
            {
                "id": f"call_web_{int(time.time()*1000)}",
                "name": "mock_web_search",
                "args": {"city": city},
            }
        ],
    }


def _manual_tool_executor(
    state: TravelState,
    knowledge_store: LocalKnowledgeStore | None = None,
) -> dict[str, Any]:
    """Distinction 1: Run raw tool calls by hand — no ToolNode, no abstractions."""
    store = knowledge_store or LocalKnowledgeStore()
    pending = state.get("pending_tool_calls", [])
    updates, messages, trace, errors = manual_execute_tool_calls(pending, store)
    return {
        **updates,
        "messages": messages,
        "tool_trace": trace,
        "errors": errors,
    }


def _reuse_context(state: TravelState) -> dict[str, Any]:
    """Distinction 3: Memory reuse — we already know the city, just refresh weather."""
    city = state.get("city", "Current destination")
    return {
        "route_reason": f"Memory Checkpoint preserved destination context for '{city}'; triggering weather refresh branch only.",
        "errors": [],
    }


def _fetch_weather(state: TravelState) -> dict[str, Any]:
    """Distinction 2, Branch 1: Grab the 7-day forecast.
    Runs in parallel with _fetch_images — they don't depend on each other."""
    try:
        city = state.get("city", "Unknown destination")
        query = (state.get("user_query") or "").lower()
        # Shift the forecast window if the user asked about 'next week' or 'tomorrow'
        start_offset = 7 if "next week" in query else (1 if "tomorrow" in query else 0)
        forecast = mock_weather_forecast(city, days=7, start_offset=start_offset)
        return {"weather_forecast": [p.model_dump(mode="json") for p in forecast]}
    except Exception as exc:
        return {"weather_forecast": [], "errors": [f"Weather service unavailable: {exc}"]}


def _fetch_images(state: TravelState) -> dict[str, Any]:
    """Distinction 2, Branch 2: Grab location photos.
    Runs in parallel with _fetch_weather — they converge at finalize."""
    try:
        city = state.get("city", "Unknown destination")
        urls = mock_image_search(city)
        return {"image_urls": urls}
    except Exception as exc:
        return {"image_urls": [], "errors": [f"Image service unavailable: {exc}"]}


def _deduplicate_errors(values: list[Any]) -> list[str]:
    """Remove duplicate error messages — no point showing 'API failed' three times."""
    result: list[str] = []
    for val in values:
        msg = str(val).strip()
        if msg and msg not in result:
            result.append(msg)
    return result


def _synthesize_llm_summary(
    city: str,
    base_summary: str,
    highlights: list[str],
    travel_notes: list[str],
    target_language: str = "English",
) -> str:
    """Hit the LLM to rewrite the summary in the target language with some flair.
    Falls back silently to the base summary if the API is down or disabled."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://aicredits.in/v1").strip()
    model_name = os.getenv("OPENAI_MODEL", "openai/gpt-5.6-luna").strip()
    use_model = os.getenv("TRAVEL_USE_MODEL", "1").strip()

    if not api_key or use_model != "1":
        return base_summary

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=0.3,
            timeout=12,
            max_retries=1,
        )

        prompt = (
            f"You are a master luxury travel guide writer. Synthesize an inspiring, evocative, and grounded 2-paragraph "
            f"travel overview for {city}.\n\n"
            f"Target Language: {target_language} (Write the entire output naturally in {target_language})\n\n"
            f"Core Facts:\n{base_summary}\n\n"
            f"Key Highlights: {', '.join(highlights[:4])}\n"
            f"Local Tips: {' '.join(travel_notes[:2])}\n\n"
            f"Output ONLY the curated summary text in clean prose without markdown headers."
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 40:
            return content.strip()
    except Exception:
        pass

    return base_summary


def _finalize(state: TravelState) -> dict[str, Any]:
    """Last node: assemble everything into a TravelResponse and ship it to the UI.
    This is where weather, images, knowledge, and packing list all come together."""
    raw_errors = state.get("errors") or []
    all_errors = raw_errors if isinstance(raw_errors, list) else [raw_errors]
    try:
        error_start = max(0, int(state.get("error_start_index", 0)))
    except (TypeError, ValueError):
        error_start = 0
    # Only surface errors from THIS run, not leftover errors from previous turns
    run_errors = list(all_errors)[error_start:]

    target_lang = state.get("target_language", "English") or "English"

    # Try to build CityKnowledge from local store first, then web search
    knowledge: CityKnowledge | None = None
    if state.get("knowledge"):
        try:
            knowledge = CityKnowledge.model_validate(state["knowledge"])
        except Exception:
            run_errors.append("Local knowledge parsing error; used structured fallback.")

    if knowledge is None and state.get("web_context"):
        try:
            context = state["web_context"]
            knowledge = (
                CityKnowledge.model_validate(context)
                if isinstance(context, dict)
                else CityKnowledge.model_validate_json(str(context))
            )
        except Exception:
            run_errors.append("Web search parsing error; used structured fallback.")

    # If both sources failed, build a minimal placeholder so the UI doesn't break
    if knowledge is None:
        city_label = display_city(state.get("city", "Unknown destination"))
        knowledge = CityKnowledge(
            city=city_label,
            country="Global Destination",
            region="Discovery Route",
            summary=(
                "Please enter a city or destination name to explore."
                if state.get("empty_request")
                else f"Exploration details for {city_label} synthesized via autonomous agent routing."
            ),
            best_time="Check local seasonal forecasts",
            highlights=[f"Historic landmarks and cultural hubs in {city_label}"],
            travel_notes=["Check local transit and reservation recommendations."],
            famous_dishes=[],
            upcoming_events=[],
            neighborhoods=[],
            local_culture=LocalCulture(),
            budget_estimates=BudgetEstimate(),
            source="mock_web_search",
        )

    # Ask the LLM to rewrite the summary in the target language (skips on empty queries)
    if not state.get("empty_request"):
        final_summary = _synthesize_llm_summary(
            city=knowledge.city,
            base_summary=knowledge.summary,
            highlights=knowledge.highlights,
            travel_notes=knowledge.travel_notes,
            target_language=target_lang,
        )
    else:
        final_summary = knowledge.summary

    # Validate weather data — discard any malformed points rather than crashing
    raw_forecast = state.get("weather_forecast", [])
    valid_forecast: list[WeatherPoint] = []
    if isinstance(raw_forecast, list):
        for item in raw_forecast[:7]:
            try:
                p = item if isinstance(item, WeatherPoint) else WeatherPoint.model_validate(item)
                valid_forecast.append(p)
            except Exception:
                run_errors.append("Discarded invalid forecast data point.")

    packing_list = _generate_weather_packing_list(valid_forecast)

    # Only keep URLs that look legit — no empty strings or weird protocols
    raw_images = state.get("image_urls", [])
    valid_images: list[str] = []
    if isinstance(raw_images, list):
        for url in raw_images[:6]:
            if isinstance(url, str) and url.strip().startswith(("https://", "http://")):
                valid_images.append(url.strip())

    errors = _deduplicate_errors(run_errors)
    status = "error" if state.get("empty_request") else ("partial" if errors else "ok")

    start_t = state.get("start_time", time.perf_counter())
    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    source_label = (
        "Internal knowledge base (ChromaDB)"
        if knowledge.source == "local_vector_store"
        else "Live Web Search"
    )

    response = TravelResponse(
        city=knowledge.city,
        country=knowledge.country,
        region=knowledge.region,
        best_time=knowledge.best_time,
        city_summary=final_summary,
        weather_forecast=valid_forecast,
        image_urls=valid_images,
        famous_dishes=knowledge.famous_dishes,
        upcoming_events=knowledge.upcoming_events,
        neighborhoods=knowledge.neighborhoods,
        local_culture=knowledge.local_culture,
        budget_estimates=knowledge.budget_estimates,
        packing_essentials=packing_list,
        target_language=target_lang,
        source=source_label,
        route_reason=str(state.get("route_reason") or ""),
        highlights=knowledge.highlights,
        travel_notes=knowledge.travel_notes,
        status=status,
        errors=errors,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        execution_time_ms=elapsed_ms,
    )
    return {"result": response.model_dump(mode="json")}


def fallback_response(city: Any = "Unknown destination", errors: list[Any] | None = None) -> TravelResponse:
    """Safe fallback — when something truly unexpected blows up,
    return a valid TravelResponse so the UI still renders cleanly."""
    return TravelResponse(
        city=display_city(city),
        country="Destination Guide",
        region="Discovery Route",
        best_time="Check local conditions",
        city_summary="The assistant encountered an issue assembling destination intelligence. Please try another city query.",
        weather_forecast=[],
        image_urls=[],
        famous_dishes=[],
        upcoming_events=[],
        neighborhoods=[],
        local_culture=LocalCulture(),
        budget_estimates=BudgetEstimate(),
        packing_essentials=[],
        target_language="English",
        source="Live Web Search",
        route_reason="Safe fallback response returned after graph boundary interception.",
        status="error",
        errors=_deduplicate_errors(errors or ["An unexpected error occurred during execution."]),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        execution_time_ms=0.0,
    )


# ---------------------------------------------------------------------------
#  GRAPH ASSEMBLY — wire up all the nodes and edges
# ---------------------------------------------------------------------------

def build_graph(persist_directory: str = ".chroma"):
    """Build and compile the full LangGraph workflow.
    
    Topology: START → parse → route → [local|web|empty|reuse] → tool_executor
              → [weather + images in parallel] → finalize → END
    
    Compiled with MemorySaver so conversation context persists across turns.
    """
    knowledge_store = LocalKnowledgeStore(persist_directory=persist_directory)
    builder = StateGraph(TravelState)

    # Register all nodes
    builder.add_node("parse_request", _parse_request)
    builder.add_node("prepare_local_tool", _prepare_local_tool)
    builder.add_node("prepare_web_tool", _prepare_web_tool)
    builder.add_node(
        "manual_tool_executor",
        lambda state: _manual_tool_executor(state, knowledge_store),
    )
    builder.add_node("empty_request", _empty_request)
    builder.add_node("reuse_context", _reuse_context)

    # These two run in parallel — that's the fan-out (Distinction 2)
    builder.add_node("fetch_weather", _fetch_weather)
    builder.add_node("fetch_images", _fetch_images)

    # Everything converges here
    builder.add_node("finalize", _finalize)

    # Wire up the edges
    builder.add_edge(START, "parse_request")

    # The big routing decision — 4-way conditional edge
    builder.add_conditional_edges(
        "parse_request",
        lambda state: _route_after_parse(state, knowledge_store),
        {
            "empty": "empty_request",
            "reuse_context": "reuse_context",
            "local_tool": "prepare_local_tool",
            "web_tool": "prepare_web_tool",
        },
    )

    builder.add_edge("prepare_local_tool", "manual_tool_executor")
    builder.add_edge("prepare_web_tool", "manual_tool_executor")

    # Fan-out: tool executor fires both weather and images at the same time
    builder.add_edge("manual_tool_executor", "fetch_weather")
    builder.add_edge("manual_tool_executor", "fetch_images")

    # Memory follow-up only needs fresh weather, skip everything else
    builder.add_edge("reuse_context", "fetch_weather")

    # Fan-in: all branches converge at finalize
    builder.add_edge("empty_request", "finalize")
    builder.add_edge("fetch_weather", "finalize")
    builder.add_edge("fetch_images", "finalize")
    builder.add_edge("finalize", END)

    # MemorySaver = conversation memory across turns (Distinction 3)
    return builder.compile(checkpointer=MemorySaver())


def response_from_state(state: dict[str, Any]) -> TravelResponse:
    """Extract TravelResponse from the graph output, with fallback if things look weird."""
    if not isinstance(state, dict):
        return fallback_response(errors=["Workflow returned an invalid state shape."])

    result = state.get("result")
    try:
        if isinstance(result, TravelResponse):
            return result
        if isinstance(result, dict):
            return TravelResponse.model_validate(result)
        return fallback_response(
            city=state.get("city", "Unknown destination"),
            errors=["No structured result found in state."],
        )
    except Exception as exc:
        return fallback_response(
            city=state.get("city", "Unknown destination"),
            errors=[f"Structured validation failed: {exc}"],
        )
