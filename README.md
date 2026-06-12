# SafeRoute AI 🗺️

AI-powered safe route planner for **Semey, Kazakhstan**. Built for Track 3 — City Safety & Social Services.

Pick a travel mode (🚗 car / 👶 child / 🧳 tourist), enter start and end points — the system draws a route that avoids danger zones and explains why it's safe (in Russian).

## Tech Stack

| Layer | Tech |
| --- | --- |
| Frontend | HTML, JS, Leaflet.js |
| Backend | Python, FastAPI, Uvicorn |
| AI | OpenRouter API (OpenAI-compatible) |

## Quick Start

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Configure environment** — create `.env` in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=openrouter/owl-alpha
```

> Get your key at [openrouter.ai/keys](https://openrouter.ai/keys). Any OpenRouter model works.

**3. Run the backend** (from `backend/`):

```bash
uvicorn main:app --reload --port 8000
```

**4. Open `frontend/index.html`** in a browser.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/zones` | Danger zones for the heatmap |
| POST | `/api/route` | `{start, end, mode}` → `{route, danger_score, ai_explanation}` |

## How It Works

```text
User picks A → B + mode
       ↓
Backend generates waypoints, adjusts route to avoid danger zones
       ↓
Danger score calculated, zones analyzed
       ↓
OpenRouter AI generates a safety explanation (Russian)
       ↓
Frontend renders route + explanation on Leaflet map
```

## Route Modes

- **Car** — avoids zones with danger level > 6
- **Child** — only green zones (danger ≤ 3), well-lit busy streets
- **Tourist** — avoids red zones (danger > 6), prefers central streets

## Project Structure

```text
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── routes.py           # Route computation + API endpoints
│   ├── ai.py               # OpenRouter AI integration
│   └── data/
│       └── semey_zones.json  # Synthetic danger zone data
├── frontend/
│   ├── index.html          # Page structure
│   ├── app.js              # Map, zones, route logic
│   └── style.css           # Styling
├── requirements.txt
└── .env.example
```

## License

MIT
