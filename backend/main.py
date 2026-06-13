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

from routes import router, set_zones, Zone, set_districts, District
from ml.danger_predictor import DangerPredictor

# Загрузить переменные окружения из .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: загрузить данные зон из JSON и ML модель."""
    # Загрузка зон
    zones_path = Path(__file__).parent / "data" / "semey_zones.json"
    with open(zones_path, "r", encoding="utf-8") as f:
        raw_zones = json.load(f)
    zones = [Zone(**z) for z in raw_zones]
    set_zones(zones)
    print(f"[OK] Zagruzheno {len(zones)} zon opasnosti")

    # Загрузка ML модели
    try:
        predictor = DangerPredictor(cache_enabled=True)
        app.state.predictor = predictor
        accuracy = predictor.metrics.get('accuracy', 'N/A')
        print(f"[OK] ML модель загружена (accuracy={accuracy})")
    except FileNotFoundError as e:
        print(f"[WARNING] ML модель не найдена: {e}")
        app.state.predictor = None

    # Загрузка районов города
    districts_path = Path(__file__).parent / "data" / "districts.json"
    with open(districts_path, "r", encoding="utf-8") as f:
        raw_districts = json.load(f)
    districts = [District(**d) for d in raw_districts["districts"]]
    set_districts(districts)
    print(f"[OK] Загружено {len(districts)} районов города")

    # Передаём районы в ML predictor для привязки к реальным границам
    if app.state.predictor:
        app.state.predictor.set_districts([d.model_dump() for d in districts])
        print(f"[OK] Данные районов переданы в ML predictor")

    yield
    # Shutdown: очистка ресурсов (если нужна)


app = FastAPI(
    title="SafeRoute AI API",
    description="API для построения безопасных маршрутов в Семею, Казахстан",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — разрешить конкретные origins для безопасности
# Для продакшена заменить на реальный домен фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",      # Live Server VS Code
        "http://127.0.0.1:5500",     # Альтернативный адрес
        "http://localhost:8080",      # Альтернативный порт
        "http://127.0.0.1:8080",
        "http://localhost:3000",      # React dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключить маршруты
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
