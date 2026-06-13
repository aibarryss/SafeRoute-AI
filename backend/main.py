"""
SafeRoute AI — FastAPI сервер.
Точка входа для запуска приложения.
"""

import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes import router, set_zones, Zone, set_districts, District
from ml.danger_predictor import DangerPredictor

logger = logging.getLogger(__name__)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: загрузить данные зон из JSON и ML модель."""
 
    zones_path = Path(__file__).parent / "data" / "semey_zones.json"
    with open(zones_path, "r", encoding="utf-8") as f:
        raw_zones = json.load(f)
    zones = [Zone(**z) for z in raw_zones]
    set_zones(zones)
    logger.info("Zagruzheno %d zon opasnosti", len(zones))

   
    try:
        predictor = DangerPredictor(cache_enabled=True)
        app.state.predictor = predictor
        accuracy = predictor.metrics.get('accuracy', 'N/A')
        logger.info("ML модель загружена (accuracy=%s)", accuracy)
    except FileNotFoundError as e:
        logger.warning("ML модель не найдена: %s", e)
        app.state.predictor = None

  
    districts_path = Path(__file__).parent / "data" / "districts.json"
    with open(districts_path, "r", encoding="utf-8") as f:
        raw_districts = json.load(f)
    districts = [District(**d) for d in raw_districts["districts"]]
    set_districts(districts)
    logger.info("Загружено %d районов города", len(districts))

   
    if app.state.predictor:
        app.state.predictor.set_districts([d.model_dump() for d in districts])
        logger.info("Данные районов переданы в ML predictor")

    yield
   


app = FastAPI(
    title="SafeRoute AI API",
    description="API для построения безопасных маршрутов в Семею, Казахстан",
    version="1.0.0",
    lifespan=lifespan,
)


ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_RAW:
    ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://saferoute-ai-production-f06b.up.railway.app",
        "https://saferoute-a1.vercel.app",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "Authorization"],
)


app.include_router(router)


@app.get("/", tags=["health"])
async def health_check():
    """Health check — проверка работоспособности сервера."""
    predictor = app.state.predictor
    ml_status = "loaded" if predictor else "not_loaded"
    ml_accuracy = predictor.metrics.get('accuracy') if predictor else None

    return {
        "status": "ok",
        "service": "SafeRoute AI API",
        "version": "2.0.0",
        "zones_loaded": True,
        "ml_model": ml_status,
        "ml_accuracy": ml_accuracy,
    }
