import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

CACHE_FILE = Path(tempfile.gettempdir()) / "events_cache.json"
STATIC_DIR = Path("static")

DEFAULT_LAOS_SOURCES = [
    "https://vientianeon.com/",
    "https://tourismluangprabang.org/festivals/",
    "https://856.today/",
]

app = FastAPI(title="LaoVibe — Laos Event Scraper & Personality Matcher")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class LaosEvent(BaseModel):
    title: str = Field(description="Name of the event or festival.")
    date_text: str = "Check organiser"
    city: str = "Laos"
    venue: str = "Various venues"
    category: str = "Culture & Events"
    price: str = "Check organiser"
    description: str = ""
    vibe_tags: list[str] = []
    ideal_personality: str = "Open-minded explorers"
    source_url: str = ""


class EventList(BaseModel):
    events: list[LaosEvent]


class ScrapeRequest(BaseModel):
    custom_urls: list[str] = []
    use_google_search: bool = False
    city_focus: str = "All of Laos"


class PersonalityProfile(BaseModel):
    social_energy: str = "Ambivert"
    pace: str = "Balanced"
    interests: list[str] = ["Arts & Culture", "Food & Coffee"]
    budget: str = "Any"
    preferred_city: str = "Anywhere in Laos"
    bio: str = ""


class RecommendRequest(BaseModel):
    query: str
    personality: PersonalityProfile
    use_live_search: bool = False


class RecommendationItem(BaseModel):
    event_title: str
    match_score: int = 80
    why_it_matches: str = ""
    date_text: str = "Check organiser"
    city: str = "Laos"
    venue: str = "Various venues"
    category: str = "Events"
    price: str = "Check organiser"
    vibe_tags: list[str] = []
    source_url: str = ""


class RecommendationResponse(BaseModel):
    assistant_message: str
    recommendations: list[RecommendationItem] = []
    suggested_followups: list[str] = []


# ---------------------------------------------------------------------------
# Helpers: Gemini Client, Cache & HTML Scraping
# ---------------------------------------------------------------------------

def get_gemini_client():
    from google import genai

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY is missing. Add it in Railway -> Variables.",
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"updated_at": None, "sources": [], "events": []}


def save_cache(events: list[dict], sources: list[str]) -> dict:
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources": sources,
        "events": events,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def clean_html_to_text(html: str, max_chars: int = 18000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if len(line) > 3)
    return cleaned[:max_chars]


async def fetch_page_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return clean_html_to_text(resp.text)
    except Exception as exc:
        print(f"[Scrape Warning] Could not fetch {url}: {exc}")
        return None


def extract_events(raw_text: str, source_url: str) -> list[LaosEvent]:
    from google.genai import types

    ai_client = get_gemini_client()
    today_str = datetime.now().strftime("%B %d, %Y")
    prompt = f"""
Today is {today_str}. Extract real events, workshops, festivals, live music, markets, or cultural activities in Laos from this webpage.
Set source_url to "{source_url}".

Content:
---
{raw_text}
---
"""
    response = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EventList,
            temperature=0.1,
        ),
    )
    parsed = response.parsed or EventList.model_validate_json(response.text)
    return parsed.events


def discover_events_via_google_search(city_focus: str, user_query: str = "") -> tuple[list[LaosEvent], list[str]]:
    from google.genai import types

    ai_client = get_gemini_client()
    today_str = datetime.now().strftime("%B %d, %Y")
    focus_clause = f"specifically in {city_focus}, Laos" if city_focus and city_focus != "All of Laos" else "across Laos (Vientiane, Luang Prabang, Vang Vieng, Pakse)"
    extra_clause = f" Prioritize events related to: '{user_query}'." if user_query else ""

    grounded_resp = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Today is {today_str}. Search the web for upcoming events, workshops, markets, live music, and festivals {focus_clause}.{extra_clause}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        ),
    )

    grounded_urls: list[str] = []
    try:
        metadata = grounded_resp.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    grounded_urls.append(chunk.web.uri)
    except Exception:
        pass

    if not grounded_resp.text:
        return [], grounded_urls

    structured_resp = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Convert these Laos event notes into structured JSON events. URLs: {', '.join(grounded_urls[:8])}\n\n{grounded_resp.text}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EventList,
            temperature=0.1,
        ),
    )
    parsed = structured_resp.parsed or EventList.model_validate_json(structured_resp.text)
    return parsed.events, grounded_urls


def deduplicate_events(events: list[LaosEvent]) -> list[dict]:
    seen = set()
    unique: list[dict] = []
    for ev in events:
        key = (re.sub(r"\W+", "", ev.title.lower()), ev.city.strip().lower())
        if key not in seen and ev.title.strip():
            seen.add(key)
            unique.append(ev.model_dump())
    return unique


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/events")
async def get_cached_events():
    return load_cache()


@app.post("/api/scrape")
async def scrape_events(req: ScrapeRequest):
    urls = DEFAULT_LAOS_SOURCES + [u.strip() for u in req.custom_urls if u.strip().startswith("http")]
    urls = list(dict.fromkeys(urls))

    all_events: list[LaosEvent] = []
    successful_sources: list[str] = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; LaoVibeEventBot/1.0)"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as http:
        for url in urls:
            text = await fetch_page_text(http, url)
            if text:
                try:
                    extracted = extract_events(text, url)
                    all_events.extend(extracted)
                    successful_sources.append(url)
                except Exception as exc:
                    print(f"[Extract Error] {url}: {exc}")

    if req.use_google_search:
        try:
            search_events, search_urls = discover_events_via_google_search(req.city_focus)
            all_events.extend(search_events)
            successful_sources.extend(search_urls[:5])
        except Exception as exc:
            print(f"[Search Grounding Warning]: {exc}")

    unique_events = deduplicate_events(all_events)
    if not unique_events:
        raise HTTPException(
            status_code=502,
            detail="Could not extract events from sources. Check your GEMINI_API_KEY or paste a custom Laos event URL.",
        )

    return save_cache(unique_events, list(dict.fromkeys(successful_sources)))


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_events(req: RecommendRequest):
    from google.genai import types

    cache = load_cache()
    events_data = cache.get("events", [])

    if req.use_live_search:
        try:
            live_events, live_urls = discover_events_via_google_search(
                city_focus=req.personality.preferred_city if req.personality.preferred_city != "Anywhere in Laos" else "All of Laos",
                user_query=req.query,
            )
            if live_events:
                existing = [LaosEvent(**e) for e in events_data]
                merged = deduplicate_events(live_events + existing)
                cache = save_cache(merged, list(dict.fromkeys(cache.get("sources", []) + live_urls[:5])))
                events_data = cache["events"]
        except Exception as exc:
            print(f"[Live Search Warning]: {exc}")

    if not events_data:
        raise HTTPException(
            status_code=400,
            detail="No events cached yet. Click 'Scrape Laos Events' at the top first!",
        )

    ai_client = get_gemini_client()
    today_str = datetime.now().strftime("%B %d, %Y")
    p = req.personality

    prompt = f"""
You are LaoVibe, a warm Laos event concierge. Today is {today_str}.
USER PERSONALITY:
- Social Energy: {p.social_energy}
- Pace: {p.pace}
- Interests: {", ".join(p.interests)}
- Budget: {p.budget}
- Preferred City: {p.preferred_city}
- Bio: "{p.bio or 'Not specified'}"

USER QUESTION: "{req.query}"

SCRAPED EVENTS:
{json.dumps(events_data[:40], ensure_ascii=False)}

Select the top 3-5 events from the list above that best match their question and personality, score 0-100, and explain why each matches. Never invent events.
"""
    response = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendationResponse,
            temperature=0.35,
        ),
    )
    return response.parsed or RecommendationResponse.model_validate_json(response.text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
