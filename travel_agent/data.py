"""All the hardcoded city data, mock API functions, and web search simulations.
Paris, Tokyo, New York live here — plus the mock weather/image/web-search functions
that stand in for real APIs."""

from __future__ import annotations

import re
from datetime import date, timedelta
from time import sleep
from typing import Any

from .models import (
    BudgetEstimate,
    CityKnowledge,
    FamousDish,
    LocalCulture,
    LocalEvent,
    Neighborhood,
    WeatherPoint,
)


CITY_FACTS: dict[str, CityKnowledge] = {
    "paris": CityKnowledge(
        city="Paris",
        country="France",
        region="Île-de-France",
        summary=(
            "Paris is a world-renowned cultural and artistic capital defined by its Haussmannian boulevards, "
            "historic bridges over the Seine, world-class museums, and vibrant sidewalk bistro culture. "
            "The city effortlessly blends iconic monuments like the Eiffel Tower and Notre-Dame with intimate "
            "neighborhood charms in Montmartre, Le Marais, and Saint-Germain-des-Prés."
        ),
        best_time="April to June or September to October (mild weather, fewer crowds)",
        highlights=[
            "The Louvre Museum & Jardin des Tuileries",
            "Montmartre & Sacré-Cœur at golden hour",
            "Seine River sunset promenade & Pont Alexandre III",
            "Sainte-Chapelle with 13th-century stained glass",
            "Historic bistro hopping in Le Marais",
        ],
        travel_notes=[
            "The Paris Métro and RER trains provide comprehensive coverage; grab a Navigo Découverte pass.",
            "Book timed-entry reservations well in advance for the Louvre, Musée d'Orsay, and the Eiffel Tower.",
            "Always greet shopkeepers and waitstaff with a polite 'Bonjour' before ordering or asking questions.",
            "Explore by arrondissements: balance high-energy museum visits with relaxed park afternoons.",
        ],
        famous_dishes=[
            FamousDish(
                name="Croissant au Beurre",
                local_name="Croissant au Beurre",
                category="Pastry & Breakfast",
                price_tier="$",
                description="Golden, multi-layered flaky viennoiserie made with high-fat Normandy butter, baked fresh every morning.",
                must_try_spot="Du Pain et des Idées (10th arr.) or local artisanal boulangeries",
            ),
            FamousDish(
                name="Boeuf Bourguignon",
                local_name="Bœuf Bourguignon",
                category="Classic Bistro",
                price_tier="$$$",
                description="Slow-braised beef chuck in rich red Burgundy wine with pearl onions, carrots, lardons, and button mushrooms.",
                must_try_spot="Chez René or Café des Musées in Le Marais",
            ),
            FamousDish(
                name="Artisanal Macarons",
                local_name="Macarons Parisiens",
                category="Dessert",
                price_tier="$$",
                description="Delicate almond meringue shells filled with rich ganache, fruit confit, or pistachio cream.",
                must_try_spot="Pierre Hermé or Ladurée Champs-Élysées",
            ),
            FamousDish(
                name="French Onion Soup",
                local_name="Soupe à l'Oignon Gratinée",
                category="Comfort Food",
                price_tier="$$",
                description="Caramelized onions in deep beef broth topped with crusty baguette slices and a thick blanket of melted Gruyère cheese.",
                must_try_spot="Au Pied de Cochon in Les Halles (open 24/7)",
            ),
        ],
        upcoming_events=[
            LocalEvent(
                title="Nuit Blanche (White Night)",
                season_or_date="Early October / Annual",
                category="Music & Arts",
                description="An all-night arts festival transforming public monuments, metro stations, and riverbanks into contemporary art installations.",
            ),
            LocalEvent(
                title="Fête de la Musique",
                season_or_date="June 21 (Summer Solstice)",
                category="Music & Arts",
                description="A nationwide celebration where free live concerts of every genre take over every square, corner, and park in Paris.",
            ),
            LocalEvent(
                title="Salon du Chocolat",
                season_or_date="Late October to Early November",
                category="Food & Wine",
                description="The world's premier chocolate and cocoa trade fair featuring master chocolatiers, tasting pavilions, and chocolate fashion shows.",
            ),
        ],
        neighborhoods=[
            Neighborhood(name="Le Marais (3rd/4th arr.)", vibe="Historic, Chic & Fashionable", best_for="Boutique shopping, art galleries, falafel & nightlife"),
            Neighborhood(name="Montmartre (18th arr.)", vibe="Bohemian & Romantic", best_for="Cobblestone walks, Sacré-Cœur views & street painters"),
            Neighborhood(name="Saint-Germain-des-Prés (6th arr.)", vibe="Literary & Classic Intellectual", best_for="Café de Flore, antique shops & luxury dining"),
            Neighborhood(name="Canal Saint-Martin (10th arr.)", vibe="Youthful, Hipster & Relaxed", best_for="Waterside picnics, natural wine bars & indie coffee"),
        ],
        local_culture=LocalCulture(
            greeting="Bonjour / Bonsoir",
            greeting_phonetic="bohn-ZHOOR / bohn-SWAHR",
            language="French (Français)",
            currency="Euro",
            currency_code="EUR (€)",
            tipping_etiquette="Service is included by law (service compris). Rounding up 1-2€ at bistros or 5-10% at fine dining is appreciated.",
            emergency_number="112 (EU General) / 15 (SAMU Medical)",
        ),
        budget_estimates=BudgetEstimate(
            backpacker_usd=65,
            moderate_usd=180,
            luxury_usd=480,
        ),
        source="local_vector_store",
    ),
    "tokyo": CityKnowledge(
        city="Tokyo",
        country="Japan",
        region="Kantō",
        summary=(
            "Tokyo is a hyper-modern metropolis that seamlessly balances centuries of heritage with cutting-edge "
            "innovation. From ancient Shinto shrines nestled within serene cedar forests to neon-lit skyscrapers "
            "and world-leading culinary craftsmanship, Tokyo offers an exceptionally clean, safe, and exhilarating "
            "urban travel experience."
        ),
        best_time="March to May (cherry blossoms) or October to November (autumn foliage)",
        highlights=[
            "Sensō-ji Temple & historic Nakamise-dori in Asakusa",
            "Shibuya Crossing & panoramic views from Shibuya Sky",
            "Meiji Jingu Shrine and the stylish backstreets of Harajuku",
            "Tsukiji Outer Market for fresh sushi and street bites",
            "Shinjuku Gyoen National Garden & Golden Gai alleyways",
        ],
        travel_notes=[
            "Use a digital Suica or Pasmo IC card on your smartphone for effortless subway and convenience store payments.",
            "Tipping is not customary in Japan; exceptional service is already standard practice.",
            "Trains and subways run with razor-sharp punctuality but stop running around midnight.",
            "Carry a small trash bag as public waste bins are rare; recycle sorting is strictly observed.",
        ],
        famous_dishes=[
            FamousDish(
                name="Edomae Sushi",
                local_name="江戸前寿司",
                category="Fine Dining / Seafood",
                price_tier="$$$$",
                description="Master-crafted nigiri sushi utilizing seasonal fish from Tokyo Bay seasoned with seasoned red vinegar sushi rice.",
                must_try_spot="Tsukiji Outer Market sushi bars or Ginza omakase counters",
            ),
            FamousDish(
                name="Rich Tonkotsu & Tsukemen Ramen",
                local_name="豚骨ラーメン / つけ麺",
                category="Street Food & Noodles",
                price_tier="$",
                description="Thick springy noodles dipped into an ultra-concentrated pork and bonito dipping broth, topped with chashu and ajitama egg.",
                must_try_spot="Tokyo Ramen Street in Tokyo Station or Rokurinsha",
            ),
            FamousDish(
                name="A5 Wagyu Sukiyaki / Yakiniku",
                local_name="和牛すき焼き",
                category="Specialty Beef",
                price_tier="$$$$",
                description="Melt-in-your-mouth marbled Japanese beef simmered in sweet soy mirin broth and dipped in fresh pasteurized egg.",
                must_try_spot="Ningyocho Imahan (established 1895)",
            ),
            FamousDish(
                name="Yakitori & Highball",
                local_name="焼き鳥",
                category="Izakaya Casual",
                price_tier="$$",
                description="Charcoal-grilled skewers of chicken over binchotan coals seasoned with tare sauce or coarse sea salt.",
                must_try_spot="Omoide Yokocho (Memory Lane) in Shinjuku or Torikizoku",
            ),
        ],
        upcoming_events=[
            LocalEvent(
                title="Sanja Matsuri (Asakusa Shrine Festival)",
                season_or_date="Third Weekend of May / Annual",
                category="Heritage & Festival",
                description="One of Tokyo's largest Shinto festivals, drawing 2 million spectators to watch 100 portable shrines (mikoshi) paraded through Asakusa.",
            ),
            LocalEvent(
                title="Sumida River Fireworks Festival",
                season_or_date="Last Saturday of July",
                category="Cultural & Seasonal",
                description="Japan's oldest fireworks display lighting up the Tokyo Skytree skyline with over 20,000 fireworks.",
            ),
            LocalEvent(
                title="Tokyo Autumn Leaves & Garden Illuminations",
                season_or_date="Mid-November to Early December",
                category="Seasonal",
                description="Rikugien and Meiji Jingu Gaien Ginkgo Avenue illuminated under dramatic night lights.",
            ),
        ],
        neighborhoods=[
            Neighborhood(name="Shibuya & Harajuku", vibe="High-Energy, Pop Culture & Fashion", best_for="Shibuya Crossing, boutique shopping & youth culture"),
            Neighborhood(name="Asakusa & Ueno", vibe="Traditional & Old-Town Nostalgia (Shitamachi)", best_for="Temples, historic craft shops, museums & street food"),
            Neighborhood(name="Ginza", vibe="Upscale Luxury & Culinary Excellence", best_for="Flagship designer boutiques, department store food halls & Michelin dining"),
            Neighborhood(name="Shinjuku & Golden Gai", vibe="Electric Metropolis & Nightlife", best_for="Skyscraper views, neon alleys, Izakayas & nightlife"),
        ],
        local_culture=LocalCulture(
            greeting="Konnichiwa / Arigatou Gozaimasu",
            greeting_phonetic="kohn-NEE-chee-wah / ah-REE-gah-toe go-ZYE-mahs",
            language="Japanese (日本語)",
            currency="Japanese Yen",
            currency_code="JPY (¥)",
            tipping_etiquette="No tipping allowed. Leaving extra money will cause the waiter to chase you down to return your change.",
            emergency_number="110 (Police) / 119 (Ambulance & Fire)",
        ),
        budget_estimates=BudgetEstimate(
            backpacker_usd=55,
            moderate_usd=160,
            luxury_usd=450,
        ),
        source="local_vector_store",
    ),
    "new york": CityKnowledge(
        city="New York",
        country="United States",
        region="New York State",
        summary=(
            "New York City is a global powerhouse of commerce, culture, gastronomy, and the arts. Spanning five "
            "dynamic boroughs, NYC captivates travelers with its iconic skyline, world-class Broadway theaters, "
            "expansive urban parks like Central Park, and an unrivaled mosaic of diverse international neighborhoods."
        ),
        best_time="April to June or September to early November (crisp, pleasant walking weather)",
        highlights=[
            "Central Park, Museum Mile & The Metropolitan Museum of Art",
            "The High Line elevated park & Hudson Yards Vessel",
            "Brooklyn Bridge stroll leading to DUMBO waterfront skyline views",
            "Broadway theater district and Times Square evening energy",
            "Culinary explorations through Greenwich Village, Chinatown & Little Italy",
        ],
        travel_notes=[
            "The NYC Subway operates 24/7; use OMNY contactless tap-to-pay at turnstiles with any credit card or phone.",
            "Comfortable, broken-in walking shoes are essential as you'll easily walk 15,000+ steps per day.",
            "Reserve popular dinner spots and Broadway tickets at least 2-4 weeks ahead.",
            "Take the free Staten Island Ferry for stunning, close-up views of the Statue of Liberty.",
        ],
        famous_dishes=[
            FamousDish(
                name="Pastrami on Rye",
                local_name="Hot Pastrami on Rye",
                category="Jewish Deli Classic",
                price_tier="$$$",
                description="Towering mound of cured, peppery, slow-smoked beef brisket sliced warm on seeded rye with spicy deli mustard.",
                must_try_spot="Katz's Delicatessen (Lower East Side) or 2nd Ave Deli",
            ),
            FamousDish(
                name="New York-Style Thin Crust Pizza",
                local_name="NY Cheese Slice",
                category="Street Food",
                price_tier="$",
                description="Wide, foldable slice with crisp bottom crust, sweet tomato sauce, and whole-milk mozzarella cheese.",
                must_try_spot="Joe's Pizza (Greenwich Village), Scarr's Pizza, or L'Industrie",
            ),
            FamousDish(
                name="Hand-Rolled Bagel with Lox & Schmear",
                local_name="Bagel with Nova & Cream Cheese",
                category="Breakfast Staple",
                price_tier="$$",
                description="Boiled-and-baked chewy bagel loaded with scallion cream cheese, Nova Scotia smoked salmon, capers, and red onion.",
                must_try_spot="Russ & Daughters (Houston St) or Ess-a-Bagel",
            ),
            FamousDish(
                name="New York Cheesecake",
                local_name="Junior's Cheesecake",
                category="Dessert",
                price_tier="$$",
                description="Ultra-dense, velvety, rich cream cheese cake on a graham cracker crust with strawberry compote.",
                must_try_spot="Junior's Restaurant (Downtown Brooklyn / Times Square)",
            ),
        ],
        upcoming_events=[
            LocalEvent(
                title="NYC Broadway Week (2-for-1 Tickets)",
                season_or_date="January & September / Bi-Annual",
                category="Music & Arts",
                description="Two-for-one ticket promotions across dozens of award-winning Broadway musicals and plays.",
            ),
            LocalEvent(
                title="Museum Mile Festival",
                season_or_date="Second Tuesday in June",
                category="Heritage & Culture",
                description="Fifth Avenue closes to traffic as 8 major museums (including The Met and Guggenheim) open their doors for free.",
            ),
            LocalEvent(
                title="Macy's Thanksgiving Day Parade & Tree Lighting",
                season_or_date="Late November to Early December",
                category="Seasonal & Holiday",
                description="Giant helium balloons parade through Manhattan followed by the Rockefeller Center Christmas Tree lighting.",
            ),
        ],
        neighborhoods=[
            Neighborhood(name="Greenwich Village & SoHo", vibe="Charming, Historic & Trendy", best_for="Brownstone walks, jazz clubs, comedy & high fashion"),
            Neighborhood(name="DUMBO & Brooklyn Heights", vibe="Waterfront, Artsy & Picturesque", best_for="Skyline views, cobblestone streets & Brooklyn Bridge park"),
            Neighborhood(name="Lower East Side & Chinatown", vibe="Gritty, Vibrant & Food-Obsessed", best_for="Dumplings, hidden speakeasies & live music venues"),
            Neighborhood(name="Upper West Side", vibe="Residential, Cultured & Green", best_for="Central Park access, Lincoln Center & iconic diners"),
        ],
        local_culture=LocalCulture(
            greeting="Hey, how's it going?",
            greeting_phonetic="Hey, how's it go-ing?",
            language="English (American)",
            currency="US Dollar",
            currency_code="USD ($)",
            tipping_etiquette="Standard tip is 18-22% at restaurants, $1-2 per drink at bars, and 15-20% for taxis/rideshares.",
            emergency_number="911 (Police, Fire, Medical)",
        ),
        budget_estimates=BudgetEstimate(
            backpacker_usd=80,
            moderate_usd=220,
            luxury_usd=550,
        ),
        source="local_vector_store",
    ),
}


CITY_ALIASES: dict[str, str] = {
    "nyc": "new york",
    "new york city": "new york",
    "the big apple": "new york",
    "manhattan": "new york",
    "paris, france": "paris",
    "city of light": "paris",
    "city of lights": "paris",
    "tokyo, japan": "tokyo",
    "edo": "tokyo",
}


IMAGE_URLS: dict[str, list[str]] = {
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1508057198894-247b23fe5ade?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1520939817895-060bdef4ad1b?auto=format&fit=crop&w=1200&q=85",
    ],
    "tokyo": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1513407030348-c983a97b98d8?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=1200&q=85",
    ],
    "new york": [
        "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1522083165195-3424ed129620?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1518391846015-55a9cc003b25?auto=format&fit=crop&w=1200&q=85",
    ],
    "kyoto": [
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1528164344705-475426879c0d?auto=format&fit=crop&w=1200&q=85",
    ],
    "snohomish": [
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1511497584788-87676104235f?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=1200&q=85",
    ],
    "rome": [
        "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1515542622106-78bda8ba0e5b?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1531572753322-ad063cecc140?auto=format&fit=crop&w=1200&q=85",
    ],
    "london": [
        "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1526129318478-62ed807ebdf9?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1486299267070-83823f5448dd?auto=format&fit=crop&w=1200&q=85",
    ],
    "barcelona": [
        "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1583422409516-2895a77efded?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1564221710304-0b34c0931a54?auto=format&fit=crop&w=1200&q=85",
    ],
}


KNOWN_CITY_BASELINES: dict[str, tuple[int, list[str]]] = {
    "paris": (19, ["Sunny", "Partly Cloudy", "Mild & Clear", "Light Showers", "Sunny", "Clear", "Partly Cloudy"]),
    "tokyo": (23, ["Clear & Bright", "Partly Cloudy", "Warm & Humid", "Scattered Showers", "Sunny", "Clear", "Pleasant Breeze"]),
    "new york": (21, ["Sunny", "Partly Cloudy", "Clear Skies", "Breezy", "Passing Showers", "Sunny", "Mild"]),
    "kyoto": (22, ["Sunny & Warm", "Partly Cloudy", "Pleasant", "Light Rain", "Clear", "Sunny", "Partly Cloudy"]),
    "snohomish": (16, ["Pacific Mist", "Partly Sunny", "Evergreen Breeze", "Light Drizzle", "Partly Cloudy", "Crisp & Clear", "Cool"]),
    "rome": (26, ["Sunny & Warm", "Mediterranean Sun", "Clear", "Warm", "Partly Cloudy", "Sunny", "Sunny"]),
    "london": (17, ["Partly Cloudy", "Overcast", "Light Rain", "Mild Breeze", "Clear Spells", "Passing Clouds", "Partly Cloudy"]),
    "barcelona": (25, ["Sunny & Breezy", "Clear Blue", "Warm", "Partly Cloudy", "Sunny", "Clear", "Coastal Sun"]),
}


def normalize_city(value: Any) -> str:
    """Turn messy user input into a clean lowercase key we can look up."""
    cleaned = " ".join(str(value or "").lower().strip().split())
    cleaned = re.sub(r"[^\w\s,-]", "", cleaned)
    return CITY_ALIASES.get(cleaned, cleaned)


def display_city(value: Any) -> str:
    """Get a nice capitalized city name for the UI."""
    normalized = normalize_city(value)
    if not normalized or normalized in ("unknown destination", "unknown"):
        return "Unknown destination"
    if normalized in CITY_FACTS:
        return CITY_FACTS[normalized].city
    return " ".join(part.capitalize() for part in normalized.split())


def mock_web_search(city: str) -> dict[str, Any]:
    """Fake web search — returns rich structured data for cities not in ChromaDB.
    Simulates real-world latency with a short sleep."""
    sleep(0.2)
    normalized = normalize_city(city) or "unknown destination"
    label = display_city(city)

    # Sensible defaults for cities we don't have custom data for
    country = "International Destination"
    region = "Regional Discovery"
    best_time = "May to September (Recommended travel window)"
    summary = (
        f"{label} is a fascinating destination discovered via live web search. Known for its distinct local character, "
        f"authentic culinary scene, and inviting neighborhood quarters, {label} offers travelers an enriching escape "
        f"with plenty of scenic walks, historic sites, and cultural experiences."
    )
    highlights = [
        f"Historic Old Town and architectural landmarks of {label}",
        f"Authentic local food markets and regional dining specialties",
        f"Scenic viewpoints and panoramic waterfront/hillside walks",
        f"Cultural museums and craft artisan workshops in {label}",
    ]
    travel_notes = [
        f"Local transit and walking are the best ways to experience {label}.",
        "Check seasonal opening times for heritage attractions and local markets.",
        "Synthesized dynamically via the Web Search routing path for destinations outside the internal vector catalog.",
    ]
    famous_dishes = [
        FamousDish(
            name=f"Traditional {label} Regional Specialty",
            category="Local Delicacy",
            price_tier="$$",
            description=f"A celebrated regional dish reflecting the agricultural and culinary heritage of {label}.",
            must_try_spot="Central food market & traditional family-run eateries",
        ),
        FamousDish(
            name="Artisanal Street Bites",
            category="Street Food",
            price_tier="$",
            description=f"Freshly prepared snacks and street delicacies popular with locals across {label}.",
            must_try_spot="Evening food stalls & town squares",
        ),
    ]
    upcoming_events = [
        LocalEvent(
            title=f"{label} Cultural Heritage & Food Festival",
            season_or_date="Summer / Annual",
            category="Heritage & Culture",
            description=f"A lively regional festival celebrating the music, arts, gastronomy, and traditions of {label}.",
        )
    ]
    neighborhoods = [
        Neighborhood(name="Historic City Center", vibe="Charming & Historic", best_for="Landmarks, architecture, cafes & walking tours"),
        Neighborhood(name="Artisan Quarter", vibe="Bohemian & Creative", best_for="Craft workshops, local dining & evening strolls"),
    ]
    local_culture = LocalCulture(
        greeting="Hello / Greetings",
        greeting_phonetic="Hello",
        language="Local Language",
        currency="Local Currency",
        currency_code="LOC",
        tipping_etiquette="Tipping customary based on local standards (typically 5-10% for good service).",
        emergency_number="112 (International Emergency)",
    )
    budget_estimates = BudgetEstimate(
        backpacker_usd=50,
        moderate_usd=140,
        luxury_usd=380,
    )

    if normalized == "snohomish":
        country = "United States"
        region = "Washington (Pacific Northwest)"
        best_time = "June through September (warm, dry Pacific Northwest summer)"
        summary = (
            "Snohomish is a charming historic river town in Western Washington, known as the 'Antique Capital of the "
            "Pacific Northwest'. Nestled along the Snohomish River with views of the Cascade Mountains, it features "
            "brick-paved streets, historic Victorian homes, artisan bakeries, and access to the Centennial Trail."
        )
        highlights = [
            "First Street Antique Mall & Historic Downtown Snohomish",
            "Snohomish River Trail & picturesque Riverview Sanctuary",
            "Centennial Trail for scenic biking and walking",
            "Local farm visits and berry picking along the Snohomish Valley",
        ]
        travel_notes = [
            "Located approximately 40 minutes north of Seattle; rental car or local transit recommended.",
            "Weekends are vibrant with vintage shoppers and farm stand visitors.",
            "Pack layers and light waterproof gear for typical Pacific Northwest weather.",
        ]
        famous_dishes = [
            FamousDish(
                name="Wild PNW Blackberry Pie",
                category="Bakery & Dessert",
                price_tier="$",
                description="Warm, flakey double-crust pie stuffed with freshly harvested wild Cascade blackberries.",
                must_try_spot="Snohomish Bakery at First & Union or Grain Artisan Bakery",
            ),
            FamousDish(
                name="Cedar-Plank Smoked Salmon",
                category="Pacific Northwest Seafood",
                price_tier="$$$",
                description="Locally caught Pacific King Salmon slow-smoked over aromatic western red cedar planks with maple glaze.",
                must_try_spot="Local riverfront grills and seasonal farm-to-table diners",
            ),
        ]
        upcoming_events = [
            LocalEvent(
                title="Snohomish Kla Ha Ya Days Festival",
                season_or_date="Mid-July / Annual",
                category="Community & Heritage",
                description="Historic community celebration with river races, street fairs, car shows, and carnival parades.",
            )
        ]
        neighborhoods = [
            Neighborhood(name="Historic Downtown First Street", vibe="Vintage & Picturesque", best_for="Antique shopping, river walks & bakeries"),
            Neighborhood(name="Snohomish River Valley", vibe="Pastoral & Scenic", best_for="Farm stands, pumpkin patches & mountain views"),
        ]
        local_culture = LocalCulture(
            greeting="Hello / Howdy",
            greeting_phonetic="Hello",
            language="English",
            currency="US Dollar",
            currency_code="USD ($)",
            tipping_etiquette="18-20% standard tipping at restaurants and cafes.",
            emergency_number="911",
        )
        budget_estimates = BudgetEstimate(backpacker_usd=60, moderate_usd=150, luxury_usd=350)

    elif normalized == "kyoto":
        country = "Japan"
        region = "Kansai"
        best_time = "March to May or October to November"
        summary = (
            "Kyoto, the ancient imperial capital of Japan, is home to thousands of classical Buddhist temples, "
            "sublime Shinto shrines, traditional wooden machiya houses, and tranquil Zen rock gardens. It represents "
            "the spiritual heart of Japanese traditional culture and seasonal aesthetics."
        )
        highlights = [
            "Fushimi Inari-taisha with thousands of vermilion torii gates",
            "Arashiyama Bamboo Grove & Tenryu-ji Zen garden",
            "Kinkaku-ji (The Golden Pavilion) reflected over its mirror pond",
            "Gion historic geisha district & traditional teahouses",
        ]
        travel_notes = [
            "Kyoto City Bus and subway networks are easy to use with standard IC cards.",
            "Visit top shrines early in the morning (around 7:00 AM) to experience serene contemplation without crowds.",
            "Dress respectfully when entering active temple sanctuaries and shrines.",
        ]
        famous_dishes = [
            FamousDish(
                name="Kyoto Kaiseki Ryori",
                local_name="懐石料理",
                category="Multi-Course Haute Cuisine",
                price_tier="$$$$",
                description="Meticulously crafted multi-course banquet celebrating seasonal ingredients, delicate broths, and artistic ceramic plating.",
                must_try_spot="Gion ryotei or Kikunoi Honten",
            ),
            FamousDish(
                name="Uji Matcha Soft Serve & Parfait",
                local_name="宇治抹茶パフェ",
                category="Dessert & Sweets",
                price_tier="$",
                description="Creamy soft-serve made from ceremonial-grade Uji green tea paired with mochi dango and sweet red azuki beans.",
                must_try_spot="Tsujiri Gion or Nakamura Tokichi",
            ),
        ]
        upcoming_events = [
            LocalEvent(
                title="Gion Matsuri (Yasaka Shrine)",
                season_or_date="Entire Month of July (Main Parade July 17)",
                category="Heritage & Culture",
                description="One of Japan's most famous festivals, featuring towering 25-meter wooden floats (Yamaboko) paraded through central Kyoto.",
            )
        ]
        neighborhoods = [
            Neighborhood(name="Gion & Higashiyama", vibe="Traditional & Historic", best_for="Geisha teahouses, wooden machiya, Kiyomizu-dera & pottery shops"),
            Neighborhood(name="Arashiyama", vibe="Scenic & Serene", best_for="Bamboo groves, monkey park, Zen temples & scenic riverboats"),
        ]
        local_culture = LocalCulture(
            greeting="Konnichiwa / Okini (Kyoto Dialect)",
            greeting_phonetic="kohn-NEE-chee-wah / oh-KEE-nee",
            language="Japanese",
            currency="Japanese Yen",
            currency_code="JPY (¥)",
            tipping_etiquette="No tipping allowed. Exceptional hospitality (Omotenashi) is included.",
            emergency_number="110 (Police) / 119 (Ambulance)",
        )
        budget_estimates = BudgetEstimate(backpacker_usd=55, moderate_usd=165, luxury_usd=460)

    elif normalized == "rome":
        country = "Italy"
        region = "Lazio"
        best_time = "April to June or September to October"
        summary = (
            "Rome, the Eternal City, is a living museum where ancient Roman ruins like the Colosseum and Forum stand "
            "beside Renaissance palaces, Baroque fountains, and vibrant piazzas filled with trattorias and espresso bars."
        )
        highlights = [
            "The Colosseum & Roman Forum archaeological area",
            "Vatican City: St. Peter's Basilica & Sistine Chapel",
            "Trevi Fountain & Spanish Steps evening stroll",
            "Trastevere neighborhood for authentic Roman pasta",
        ]
        travel_notes = [
            "Carry a refillable water bottle to drink from Rome's historic public fountains ('nasoni').",
            "Pre-purchase skip-the-line tickets for the Vatican Museums and Colosseum.",
            "Always validate public transit tickets before boarding buses or trams.",
        ]
        famous_dishes = [
            FamousDish(
                name="Authentic Carbonara & Cacio e Pepe",
                local_name="Pasta alla Carbonara",
                category="Classic Pasta",
                price_tier="$$",
                description="Al dente rigatoni coated in crispy guanciale, pecorino romano cheese, farm-fresh egg yolk, and cracked black pepper (no cream!).",
                must_try_spot="Da Enzo al 29 (Trastevere) or Roscioli",
            ),
            FamousDish(
                name="Artisanal Gelato",
                local_name="Gelato Artigianale",
                category="Dessert",
                price_tier="$",
                description="Dense, slow-churned Italian ice cream made with Bronte pistachios, Piedmont hazelnuts, and dark chocolate.",
                must_try_spot="Frigidarium or Giolitti",
            ),
        ]
        upcoming_events = [
            LocalEvent(
                title="Festa de' Noantri (Trastevere)",
                season_or_date="Late July / Annual",
                category="Cultural & Religious",
                description="Historic celebration of the Madonna del Carmine with a grand boat procession along the Tiber River and street fireworks.",
            )
        ]
        neighborhoods = [
            Neighborhood(name="Trastevere", vibe="Bohemian & Lively", best_for="Cobblestone alleys, authentic trattorias & evening wine bars"),
            Neighborhood(name="Centro Storico & Campo de' Fiori", vibe="Grand & Historic", best_for="Piazzas, Pantheon, street markets & espresso bars"),
        ]
        local_culture = LocalCulture(
            greeting="Ciao / Buongiorno",
            greeting_phonetic="CHOW / bwon-ZHOR-noh",
            language="Italian (Italiano)",
            currency="Euro",
            currency_code="EUR (€)",
            tipping_etiquette="Cover charge (coperto) is usually on the bill. Leaving 1-2€ per person for dinner is polite.",
            emergency_number="112",
        )
        budget_estimates = BudgetEstimate(backpacker_usd=60, moderate_usd=170, luxury_usd=440)

    return {
        "city": label,
        "country": country,
        "region": region,
        "summary": summary,
        "best_time": best_time,
        "highlights": highlights,
        "travel_notes": travel_notes,
        "famous_dishes": [d.model_dump(mode="json") for d in famous_dishes],
        "upcoming_events": [e.model_dump(mode="json") for e in upcoming_events],
        "neighborhoods": [n.model_dump(mode="json") for n in neighborhoods],
        "local_culture": local_culture.model_dump(mode="json"),
        "budget_estimates": budget_estimates.model_dump(mode="json"),
        "source": "mock_web_search",
    }


def mock_weather_forecast(city: str, days: int = 7, start_offset: int = 0) -> list[WeatherPoint]:
    """Generate a fake but realistic 7-day forecast — deterministic per city so results are consistent."""
    sleep(0.15)
    normalized = normalize_city(city) or "unknown destination"
    try:
        requested_days = int(days)
    except (TypeError, ValueError):
        requested_days = 7
    days = max(1, min(requested_days, 7))

    baseline_info = KNOWN_CITY_BASELINES.get(
        normalized,
        (20, ["Sunny", "Partly Cloudy", "Clear", "Mild Showers", "Sunny", "Pleasant", "Breezy"]),
    )
    baseline_temp, condition_pool = baseline_info

    # Use a hash of the city name so each city gets unique but repeatable temps
    city_hash = sum(ord(c) for c in normalized)
    temp_offset = (city_hash % 7) - 3

    today = date.today() + timedelta(days=start_offset)
    points: list[WeatherPoint] = []

    for idx in range(days):
        current_date = today + timedelta(days=idx)
        daily_variation = ((idx * 3 + city_hash) % 5) - 2
        temp_c = float(baseline_temp + temp_offset + daily_variation)
        precip_pct = int(((idx * 19 + city_hash * 7) % 85))
        humidity_pct = int(45 + ((idx * 13 + city_hash) % 45))
        wind_speed = int(8 + ((idx * 7 + city_hash) % 22))
        cond = condition_pool[idx % len(condition_pool)]
        if precip_pct > 60 and "Rain" not in cond and "Showers" not in cond:
            cond = "Scattered Rain"

        points.append(
            WeatherPoint(
                date=current_date.isoformat(),
                day=current_date.strftime("%a, %b %d"),
                temperature_c=temp_c,
                condition=cond,
                precipitation_probability=precip_pct,
                humidity_pct=humidity_pct,
                wind_kmh=wind_speed,
            )
        )
    return points


def mock_image_search(city: str) -> list[str]:
    """Return Unsplash photo URLs for the city — falls back to generic travel shots."""
    sleep(0.15)
    normalized = normalize_city(city) or "unknown destination"
    if normalized in IMAGE_URLS:
        return IMAGE_URLS[normalized]

    # Generic travel photos for cities we don't have custom images for
    return [
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=85",
        "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?auto=format&fit=crop&w=1200&q=85",
    ]
