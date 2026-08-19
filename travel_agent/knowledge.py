"""ChromaDB vector store wrapper — this is where Paris, Tokyo, and New York live.
If a city is in here, we skip the web search entirely."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import CITY_FACTS, normalize_city
from .models import (
    BudgetEstimate,
    CityKnowledge,
    FamousDish,
    LocalCulture,
    LocalEvent,
    Neighborhood,
)


class LocalKnowledgeStore:
    """Two-layer lookup: exact match against our hardcoded catalog first,
    then fall back to ChromaDB semantic search if that misses.

    If Chroma itself fails to initialize (locked DB, missing deps, etc.),
    we still work — just without the vector similarity fallback."""

    def __init__(self, persist_directory: str = ".chroma") -> None:
        self._facts = CITY_FACTS
        self._client: Any | None = None
        self._collection: Any | None = None
        self.initialization_error: str | None = None

        try:
            import chromadb

            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
            self._collection = self._client.get_or_create_collection(
                name="travel_city_knowledge_v3",
                metadata={"description": "Vector store of pre-populated city facts with culinary & culture metadata"},
            )
            self._seed()
        except Exception as exc:
            # Chroma blew up — no worries, exact-match still works fine
            self._client = None
            self._collection = None
            self.initialization_error = str(exc)

    def _seed(self) -> None:
        """Push our hardcoded city data into Chroma if it's not already there.
        Only runs on first launch or after a DB wipe."""
        if self._collection is None:
            return

        try:
            existing = self._collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
            missing = [key for key in self._facts if key not in existing_ids]
            if not missing:
                return

            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []

            for key in missing:
                fact = self._facts[key]
                # Build a rich text doc so Chroma's embeddings have plenty to work with
                doc_text = (
                    f"City: {fact.city}\n"
                    f"Country: {fact.country}\n"
                    f"Region: {fact.region}\n"
                    f"Summary: {fact.summary}\n"
                    f"Best time to visit: {fact.best_time}\n"
                    f"Highlights: {', '.join(fact.highlights)}\n"
                    f"Travel notes: {' '.join(fact.travel_notes)}\n"
                    f"Famous Dishes: {', '.join(d.name for d in fact.famous_dishes)}\n"
                    f"Key Events: {', '.join(e.title for e in fact.upcoming_events)}"
                )
                documents.append(doc_text)
                # Stash structured JSON in metadata so we can reconstruct CityKnowledge on retrieval
                metadatas.append(
                    {
                        "city": fact.city,
                        "country": fact.country,
                        "region": fact.region,
                        "summary": fact.summary,
                        "best_time": fact.best_time,
                        "highlights": json.dumps(fact.highlights),
                        "travel_notes": json.dumps(fact.travel_notes),
                        "famous_dishes": json.dumps([d.model_dump(mode="json") for d in fact.famous_dishes]),
                        "upcoming_events": json.dumps([e.model_dump(mode="json") for e in fact.upcoming_events]),
                        "neighborhoods": json.dumps([n.model_dump(mode="json") for n in fact.neighborhoods]),
                        "local_culture": fact.local_culture.model_dump_json(),
                        "budget_estimates": fact.budget_estimates.model_dump_json(),
                        "source": fact.source,
                    }
                )
                ids.append(key)

            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as exc:
            self.initialization_error = f"Seeding failed: {exc}"

    def has_city(self, city: str) -> bool:
        """Quick check — do we have this city indexed? Tries exact match first,
        then asks Chroma for a semantic match within a tight distance threshold."""
        normalized = normalize_city(city)
        if normalized in self._facts:
            return True

        if self._collection is not None:
            try:
                results = self._collection.query(query_texts=[city], n_results=1)
                distances = results.get("distances", [[]])[0]
                ids = results.get("ids", [[]])[0]
                # 0.6 is pretty strict — only accept if Chroma is very confident
                if distances and distances[0] < 0.6 and ids:
                    matched_id = ids[0]
                    if matched_id in self._facts:
                        return True
            except Exception:
                pass
        return False

    def get_city(self, city: str) -> CityKnowledge | None:
        """Pull the full city knowledge — first from our in-memory cache,
        then from Chroma if the exact key doesn't match but semantics do."""
        normalized = normalize_city(city)
        fact = self._facts.get(normalized)
        if fact is not None:
            return fact.model_copy(deep=True)

        if self._collection is None:
            return None

        try:
            results = self._collection.query(query_texts=[city], n_results=1)
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Slightly looser threshold here (0.65) since we're already committed to fetching
            if metadatas and distances and distances[0] < 0.65:
                raw_meta = dict(metadatas[0])
                highlights = json.loads(raw_meta.get("highlights", "[]"))
                travel_notes = json.loads(raw_meta.get("travel_notes", "[]"))
                dishes_raw = json.loads(raw_meta.get("famous_dishes", "[]"))
                events_raw = json.loads(raw_meta.get("upcoming_events", "[]"))
                neigh_raw = json.loads(raw_meta.get("neighborhoods", "[]"))
                culture_raw = json.loads(raw_meta.get("local_culture", "{}"))
                budget_raw = json.loads(raw_meta.get("budget_estimates", "{}"))

                return CityKnowledge(
                    city=raw_meta.get("city", "Unknown"),
                    country=raw_meta.get("country", "Unknown"),
                    region=raw_meta.get("region", "Discovery Region"),
                    summary=raw_meta.get("summary", ""),
                    best_time=raw_meta.get("best_time", "Anytime"),
                    highlights=highlights,
                    travel_notes=travel_notes,
                    famous_dishes=[FamousDish.model_validate(d) for d in dishes_raw],
                    upcoming_events=[LocalEvent.model_validate(e) for e in events_raw],
                    neighborhoods=[Neighborhood.model_validate(n) for n in neigh_raw],
                    local_culture=LocalCulture.model_validate(culture_raw) if culture_raw else LocalCulture(),
                    budget_estimates=BudgetEstimate.model_validate(budget_raw) if budget_raw else BudgetEstimate(),
                    source="local_vector_store",
                )
        except Exception:
            return None

        return None

    def stats(self) -> dict[str, Any]:
        """Quick health check — useful for the sidebar debug panel."""
        return {
            "indexed_cities": list(self._facts.keys()),
            "vector_store_available": self._collection is not None,
            "initialization_error": self.initialization_error,
        }
