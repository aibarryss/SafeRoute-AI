# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SafeRoute AI** — AI-powered safe route planner for Semey, Kazakhstan. Built for Track 3 (City Safety & Social Services) of a hackathon. Users pick a travel mode (car / child / tourist), enter start and end points, and the system draws a route on a Leaflet map that avoids danger zones, with a Claude-generated explanation of why the route is safe.

All UI text and AI explanations are in Russian.

## Running the App

**Backend** (from `backend/`):

```bash
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** — open `frontend/index.html` directly in a browser, or serve it with any static server on a different port. The frontend talks to the backend via HTTP (CORS is configured in `main.py`).

## Architecture

Two decoupled layers communicating over HTTP:

```text
Frontend (HTML/JS/Leaflet)  →  Backend (FastAPI)  →  {semey_zones.json, Claude API}
```

- **Frontend** owns all map rendering, zone visualisation, and user interaction. No build step — plain JS with Leaflet.js from CDN.
- **Backend** owns route computation logic and AI explanation generation. Loads zone data from `backend/data/semey_zones.json` into memory.
- **Claude API** (`backend/ai.py`) — model: `claude-sonnet-4-6`. Receives the computed route + mode, returns a plain-language Russian explanation. Never called directly from the frontend.

## API Endpoints

- `GET /api/zones` — Returns danger zones array for the heatmap overlay.
- `POST /api/route` — Accepts `{start: {lat,lng}, end: {lat,lng}, mode}`, returns `{route, danger_score, ai_explanation}`. Mode is one of `"car"`, `"child"`, `"tourist"`.

## Danger Zone Model

Zones are circular areas defined by `{id, name, danger_level (1–10), lat, lng, radius}`. Colours on the map: 1–3 green, 4–6 yellow, 7–10 red.

Route mode logic:

- **Car** — avoids zones with danger > 6 (accidents, theft)
- **Child** — only green zones, well-lit busy streets
- **Tourist** — avoids red zones, prefers central streets

## Key Files

- `backend/main.py` — FastAPI app entry point, CORS setup.
- `backend/routes.py` — `/api/zones` and `/api/route` handlers, route computation logic.
- `backend/ai.py` — Claude API client and prompt construction.
- `backend/data/semey_zones.json` — Synthetic danger zone data for Semey.
- `frontend/app.js` — Map init, zone rendering, route requests, AI explanation display.
- `frontend/index.html` — Page structure, Leaflet CSS/JS CDN includes.
- `frontend/style.css` — Layout and UI styling.

## Data Note

Zone data is synthetic. The architecture is designed to be swapped to real sources (police data, citizen reports, city cameras) without changing the API contract.
