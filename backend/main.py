"""
SafeRoute AI — FastAPI сервер.
Точка входа для запуска приложения.
"""

import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes import router, set_zones, Zone

# Загрузить переменные окружения из .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: загрузить данные зон из JSON."""
    zones_path = Path(__file__).parent / "data" / "semey_zones.json"
    with open(zones_path, "r", encoding="utf-8") as f:
        raw_zones = json.load(f)
    zones = [Zone(**z) for z in raw_zones]
    set_zones(zones)
    print(f"[OK] Zagruzheno {len(zones)} zon opasnosti")
    yield
    # Shutdown: очистка ресурсов (если нужна)


app = FastAPI(
    title="SafeRoute AI API",
    description="API для построения безопасных маршрутов в Семею, Казахстан",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — разрешить все origins (hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключить маршруты
app.include_router(router)


@app.get("/", tags=["health"])
async def health_check():
    """Health check — проверка работоспособности сервера."""
    return {
        "status": "ok",
        "service": "SafeRoute AI API",
        "version": "1.0.0",
        "zones_loaded": True,
    }
