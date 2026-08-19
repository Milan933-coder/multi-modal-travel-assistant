"""All the Pydantic models and typed state that flows through the LangGraph pipeline."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, computed_field


class WeatherPoint(BaseModel):
    """One day's weather — the UI card grid and Plotly chart both consume this."""

    date: str
    day: str
    temperature_c: float = Field(description="Expected daytime temperature in Celsius")
    condition: str = Field(default="Clear", description="Weather condition summary")
    precipitation_probability: int = Field(default=0, ge=0, le=100, description="Chance of precipitation %")
    humidity_pct: int = Field(default=55, ge=0, le=100, description="Relative humidity %")
    wind_kmh: int = Field(default=12, ge=0, description="Average wind speed in km/h")

    @computed_field  # type: ignore[misc]
    @property
    def temperature_f(self) -> float:
        """Auto-convert to Fahrenheit so the UI toggle just reads this field directly."""
        return round((self.temperature_c * 9 / 5) + 32, 1)


class FamousDish(BaseModel):
    """A local dish worth flying for — rendered as cards in the culinary tab."""

    name: str
    local_name: str = ""
    category: str = "Signature Dish"  # e.g. Street Food, Fine Dining, Pastry, Beverage
    price_tier: Literal["$", "$$", "$$$", "$$$$"] = "$$"
    description: str
    must_try_spot: str = "Local markets & neighborhood bistros"


class LocalEvent(BaseModel):
    """Festivals, holidays, seasonal celebrations — stuff you'd regret missing."""

    title: str
    season_or_date: str
    category: str = "Cultural Festival"  # Music & Arts, Heritage, Food & Wine, etc.
    description: str


class Neighborhood(BaseModel):
    """A neighborhood worth wandering — shows up in the guide tab."""

    name: str
    vibe: str
    best_for: str


class LocalCulture(BaseModel):
    """The practical stuff: how to say hi, what currency to carry, who to tip."""

    greeting: str = "Hello"
    greeting_phonetic: str = "Hello"
    language: str = "Local Language"
    currency: str = "Local Currency"
    currency_code: str = "USD"
    tipping_etiquette: str = "Check local customs"
    emergency_number: str = "112 / 911"


class BudgetEstimate(BaseModel):
    """Daily spend per person in USD — three tiers from hostel-hopping to five-star."""

    backpacker_usd: int = 50
    moderate_usd: int = 150
    luxury_usd: int = 400


class CityKnowledge(BaseModel):
    """Everything we know about a city — either from ChromaDB or web search.
    This is the intermediate data that gets assembled into the final response."""

    city: str
    country: str
    region: str = "Discovery Region"
    summary: str
    best_time: str
    highlights: list[str] = Field(default_factory=list)
    travel_notes: list[str] = Field(default_factory=list)
    famous_dishes: list[FamousDish] = Field(default_factory=list)
    upcoming_events: list[LocalEvent] = Field(default_factory=list)
    neighborhoods: list[Neighborhood] = Field(default_factory=list)
    local_culture: LocalCulture = Field(default_factory=LocalCulture)
    budget_estimates: BudgetEstimate = Field(default_factory=BudgetEstimate)
    source: Literal["local_vector_store", "live_web_search", "mock_web_search"] = "local_vector_store"


class TravelResponse(BaseModel):
    """The big final object that the Streamlit UI actually renders.
    Every tab, card, and chart in the frontend maps to a field here."""

    city: str
    country: str
    region: str = "Regional Area"
    best_time: str
    city_summary: str
    weather_forecast: list[WeatherPoint] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    famous_dishes: list[FamousDish] = Field(default_factory=list)
    upcoming_events: list[LocalEvent] = Field(default_factory=list)
    neighborhoods: list[Neighborhood] = Field(default_factory=list)
    local_culture: LocalCulture = Field(default_factory=LocalCulture)
    budget_estimates: BudgetEstimate = Field(default_factory=BudgetEstimate)
    packing_essentials: list[str] = Field(default_factory=list)
    target_language: str = "English"
    source: Literal["Internal knowledge base (ChromaDB)", "Live Web Search", "Mock web search"] = (
        "Internal knowledge base (ChromaDB)"
    )
    route_reason: str = ""
    highlights: list[str] = Field(default_factory=list)
    travel_notes: list[str] = Field(default_factory=list)
    status: Literal["ok", "partial", "error"] = "ok"
    errors: list[str] = Field(default_factory=list)
    generated_at: str = ""
    execution_time_ms: float | None = None


class ToolCall(TypedDict, total=False):
    """Shape of a raw tool call — we build these manually instead of relying on ToolNode."""

    id: str
    name: str
    args: dict[str, Any]


class TravelState(TypedDict, total=False):
    """The shared state dict that every LangGraph node reads from and writes to.
    `Annotated[..., add]` fields accumulate across nodes instead of overwriting."""

    messages: Annotated[list[AnyMessage], add]
    user_query: str
    target_language: str
    city: str
    country: str
    request_kind: Literal["city_overview", "weather_follow_up"]
    empty_request: bool
    refresh_weather_only: bool
    knowledge: dict[str, Any] | None
    web_context: str
    weather_forecast: list[dict[str, Any]]
    image_urls: list[str]
    packing_essentials: list[str]
    pending_tool_calls: list[ToolCall]
    tool_trace: Annotated[list[dict[str, Any]], add]
    errors: Annotated[list[str], add]
    route_reason: str
    error_start_index: int
    result: dict[str, Any] | None
    start_time: float
