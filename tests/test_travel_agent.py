"""Tests for the travel agent — covers routing, memory, tool dispatch, and edge cases."""

import unittest
from unittest.mock import patch

from travel_agent.data import CITY_FACTS, display_city, mock_web_search, normalize_city
from travel_agent.graph import _generate_weather_packing_list, build_graph, fallback_response, response_from_state
from travel_agent.models import CityKnowledge, FamousDish, LocalCulture, LocalEvent, TravelResponse, WeatherPoint
from travel_agent.tools import manual_execute_tool_calls


class FakeKnowledgeStore:
    """In-memory stand-in for ChromaDB — no disk I/O, no flaky test failures."""

    def __init__(self, persist_directory=".chroma"):
        self.persist_directory = persist_directory

    def has_city(self, city):
        normalized = normalize_city(city)
        return normalized in CITY_FACTS

    def get_city(self, city):
        normalized = normalize_city(city)
        fact = CITY_FACTS.get(normalized)
        return fact.model_copy(deep=True) if fact else None

    def stats(self):
        return {"indexed_cities": list(CITY_FACTS.keys()), "vector_store_available": True}


class TravelAgentTests(unittest.TestCase):
    def setUp(self):
        self.store_patch = patch("travel_agent.graph.LocalKnowledgeStore", FakeKnowledgeStore)
        self.store_patch.start()
        self.graph = build_graph()

    def tearDown(self):
        self.store_patch.stop()

    def invoke(self, query, thread_id="test-thread", target_language="English"):
        state = self.graph.invoke(
            {"user_query": query, "target_language": target_language},
            config={"configurable": {"thread_id": thread_id}},
        )
        return response_from_state(state)

    def test_known_city_uses_local_vector_route(self):
        """Paris should hit ChromaDB, return weather, images, dishes — the whole package."""
        response = self.invoke("Tell me about Paris", thread_id="paris-thread")
        self.assertEqual(response.city, "Paris")
        self.assertIn("Internal knowledge base", response.source)
        self.assertEqual(response.status, "ok")
        self.assertEqual(len(response.weather_forecast), 7)
        self.assertTrue(len(response.image_urls) >= 3)
        self.assertTrue(len(response.highlights) > 0)
        self.assertTrue(len(response.travel_notes) > 0)
        self.assertTrue(len(response.famous_dishes) > 0)
        self.assertTrue(len(response.upcoming_events) > 0)
        self.assertTrue(len(response.packing_essentials) > 0)

    def test_city_aliases_route_correctly(self):
        """'NYC' should resolve to 'New York' and hit the vector store, not web search."""
        response = self.invoke("Explore NYC", thread_id="nyc-thread")
        self.assertEqual(response.city, "New York")
        self.assertIn("Internal knowledge base", response.source)
        self.assertTrue(len(response.famous_dishes) > 0)

    def test_unknown_city_uses_web_search_switch(self):
        """Kyoto and Snohomish aren't in ChromaDB — they should route to web search."""
        response_kyoto = self.invoke("Tell me about Kyoto", thread_id="kyoto-thread")
        self.assertEqual(response_kyoto.city, "Kyoto")
        self.assertEqual(response_kyoto.source, "Live Web Search")
        self.assertEqual(response_kyoto.status, "ok")
        self.assertTrue(len(response_kyoto.weather_forecast) > 0)
        self.assertTrue(len(response_kyoto.famous_dishes) > 0)

        response_sno = self.invoke("What about Snohomish?", thread_id="sno-thread")
        self.assertEqual(response_sno.city, "Snohomish")
        self.assertEqual(response_sno.source, "Live Web Search")
        self.assertTrue(len(response_sno.famous_dishes) > 0)

    def test_weather_follow_up_preserves_city_context(self):
        """Distinction 3: 'Tokyo' then 'what about next week?' should keep Tokyo,
        skip the full lookup, and only refresh weather data."""
        self.invoke("Tell me about Tokyo", thread_id="follow-up-thread")
        response = self.invoke("What about next week?", thread_id="follow-up-thread")
        self.assertEqual(response.city, "Tokyo")
        self.assertIn("Internal knowledge base", response.source)
        self.assertIn("weather refresh branch only", response.route_reason)
        self.assertEqual(len(response.weather_forecast), 7)
        self.assertTrue(len(response.famous_dishes) > 0)

    def test_multi_language_target_preservation(self):
        """Target language should flow through the pipeline and show up in the response."""
        response = self.invoke("Tell me about Tokyo", thread_id="lang-thread", target_language="Japanese")
        self.assertEqual(response.target_language, "Japanese")

    def test_smart_packing_list_generation(self):
        """Cold + rainy forecast should trigger umbrella and warm layer suggestions."""
        rainy_forecast = [
            WeatherPoint(
                date="2026-08-19",
                day="Wed",
                temperature_c=10.0,
                condition="Heavy Rain",
                precipitation_probability=80,
                wind_kmh=25,
            )
        ]
        packing = _generate_weather_packing_list(rainy_forecast)
        self.assertTrue(any("umbrella" in p.lower() or "jacket" in p.lower() for p in packing))
        self.assertTrue(any("merino" in p.lower() or "fleece" in p.lower() or "sweater" in p.lower() for p in packing))

    def test_empty_request_returns_structured_error(self):
        """Blank input should give back a clean error response, not a crash."""
        response = self.invoke("   ", thread_id="empty-thread")
        self.assertEqual(response.status, "error")
        self.assertEqual(response.weather_forecast, [])
        self.assertTrue(response.errors)

    def test_new_request_does_not_inherit_previous_errors(self):
        """Errors from a bad query shouldn't pollute the next valid query."""
        self.invoke("   ", thread_id="error-reset-thread")
        response = self.invoke("Paris", thread_id="error-reset-thread")
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.errors, [])

    def test_manual_tool_dispatcher_distinction_1(self):
        """Distinction 1: Valid, malformed, and unknown tool calls should all be handled
        gracefully — 3 messages, 3 trace entries, 2 errors (malformed + unknown)."""
        fake_store = FakeKnowledgeStore()
        valid_call = {
            "id": "call_1",
            "name": "lookup_local_knowledge",
            "args": {"city": "Paris"},
        }
        malformed_call = None
        unknown_call = {"id": "call_2", "name": "unknown_tool", "args": {}}

        updates, messages, trace, errors = manual_execute_tool_calls(
            [valid_call, malformed_call, unknown_call], fake_store
        )
        self.assertIn("knowledge", updates)
        self.assertEqual(len(messages), 3)
        self.assertEqual(len(trace), 3)
        self.assertEqual(len(errors), 2)
        self.assertEqual(messages[0].name, "lookup_local_knowledge")

    def test_weather_point_temperature_conversion(self):
        """Quick sanity check — 20°C should be exactly 68°F."""
        point = WeatherPoint(
            date="2026-08-19",
            day="Wed",
            temperature_c=20.0,
            condition="Sunny",
            precipitation_probability=15,
        )
        self.assertEqual(point.temperature_f, 68.0)

    def test_fallback_response_is_always_valid(self):
        """Even the emergency fallback should produce a proper Pydantic model
        with deduplicated errors."""
        response = fallback_response(city="Unknown", errors=["timeout", "timeout"])
        self.assertEqual(response.status, "error")
        self.assertEqual(response.errors, ["timeout"])


if __name__ == "__main__":
    unittest.main()
