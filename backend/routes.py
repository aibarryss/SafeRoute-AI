"""
API маршруты и логика вычисления безопасных маршрутов.
"""

import os
import math
import json
import httpx
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from fastapi import APIRouter, Request, HTTPException

TWOGIS_API_KEY = os.getenv("TWOGIS_API_KEY")


# Pydantic модели

class ModeEnum(str, Enum):
    """Допустимые режимы маршрута."""
    car = "car"
    child = "child"
    tourist = "tourist"


class Coordinate(BaseModel):
    """Географическая координата."""
    lat: float = Field(..., ge=-90, le=90, description="Широта")
    lng: float = Field(..., ge=-180, le=180, description="Долгота")


class RouteRequest(BaseModel):
    """Запрос на построение маршрута."""
    start: Coordinate
    end: Coordinate
    mode: ModeEnum
    hour: Optional[int] = Field(None, ge=0, le=23, description="Час суток (0-23) для ML-прогноза. Если не указан — текущее время.")
    day: Optional[int] = Field(None, ge=0, le=6, description="День недели (0=Пн, 6=Вс) для ML-прогноза. Если не указан — текущий день.")


class RouteResponse(BaseModel):
    """Ответ с построенным маршрутом."""
    route: List[Coordinate]
    danger_score: float = Field(..., ge=0, le=10)
    ai_explanation: str
    warnings: List[str] = Field(default_factory=list)  # Предупреждения о опасных районах
    route_buildable: bool = Field(default=True)  # Можно ли построить безопасный маршрут
    warnings: List[str] = Field(default_factory=list, description="Предупреждения о корректировке маршрута")


class Zone(BaseModel):
    """Зона опасности."""
    id: str
    name: str
    danger_level: int = Field(..., ge=1, le=10)
    lat: float
    lng: float
    radius: float = Field(..., gt=0, description="Радиус зоны в метрах")
    description: str = ""


class PredictRequest(BaseModel):
    """Запрос на ML предсказание уровня опасности."""
    lat: float = Field(..., description="Широта")
    lng: float = Field(..., description="Долгота")
    hour: int = Field(12, ge=0, le=23, description="Время суток (0-23)")
    day: int = Field(0, ge=0, le=6, description="День недели (0-6)")


class PredictResponse(BaseModel):
    """Ответ с ML предсказанием уровня опасности."""
    danger_level: int = Field(..., ge=1, le=10, description="Предсказанный уровень опасности")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность модели")
    risk_category: str = Field(..., description="Категория риска (low/medium/high/critical)")
    probabilities: Dict = Field(..., description="Вероятности для каждого уровня")
    features_used: Dict = Field(..., description="Использованные признаки")
    latency_ms: float = Field(..., description="Время выполнения в миллисекундах")


class District(BaseModel):
    """Район города с уровнем опасности."""
    id: str
    name: str
    danger_level: int = Field(..., ge=1, le=10)
    description: str = ""
    polygon: List[Dict[str, float]]  # [{"lat": ..., "lng": ...}, ...]


class DistrictUpdateRequest(BaseModel):
    """Запрос на обновление уровня опасности района."""
    danger_level: int = Field(..., ge=1, le=10, description="Новый уровень опасности (1-10)")
    description: Optional[str] = Field(None, description="Обновлённое описание района")


# Router и глобальное состояние

router = APIRouter(prefix="/api", tags=["routes"])


@router.get("/config")
async def get_config():
    """
    Возвращает публичную конфигурацию для фронтенда.
    2GIS ключ нужен для инициализации карты MapGL.
    """
    return {
        "twogis_api_key": TWOGIS_API_KEY or ""
    }

# Глобальное хранилище зон — загружается в main.py через set_zones()
_zones: List[Zone] = []
_districts: List[District] = []


def set_zones(zones: List[Zone]) -> None:
    """Устанавливает данные зон (вызывается при старте сервера)."""
    global _zones
    _zones = zones


def get_zones() -> List[Zone]:
    """Возвращает загруженные зоны."""
    return _zones


def set_districts(districts: List[District]) -> None:
    """Устанавливает данные районов (вызывается при старте сервера)."""
    global _districts
    _districts = districts


def get_districts() -> List[District]:
    """Возвращает загруженные районы."""
    return _districts


# Константы — параметры маршрутизации для каждого режима

class ModeParams:
    """Параметры маршрутизации для конкретного режима."""
    def __init__(
        self,
        danger_threshold: int,
        buffer_m: int,
        iterations: int,
        prefer_central: bool = False,
        smooth_factor: float = 0.0
    ):
        self.danger_threshold = danger_threshold  # Избегать зоны с danger > threshold
        self.buffer_m = buffer_m                  # Запас за границей зоны (метры)
        self.iterations = iterations              # Итерации для сглаживания
        self.prefer_central = prefer_central      # Предпочитать центральные улицы
        self.smooth_factor = smooth_factor        # Коэффициент сглаживания маршрута

MODE_PARAMS: Dict[str, ModeParams] = {
    "car": ModeParams(
        danger_threshold=6,
        buffer_m=80,      # Машина может объехать дальше
        iterations=3,
        prefer_central=False,
        smooth_factor=0.1  # Небольшое сглаживание
    ),
    "child": ModeParams(
        danger_threshold=3,
        buffer_m=120,     # Максимальный запас для ребёнка
        iterations=5,     # Больше итераций для гладкости
        prefer_central=False,
        smooth_factor=0.2  # Более плавный маршрут
    ),
    "tourist": ModeParams(
        danger_threshold=6,
        buffer_m=100,
        iterations=4,
        prefer_central=True,  # Туристы предпочитают центр
        smooth_factor=0.15
    ),
}

# Центр Семея для логики "предпочитать центральные улицы"
CITY_CENTER = Coordinate(lat=50.4111, lng=80.2275)
CENTRAL_RADIUS_M = 3000  # Радиус "центральной зоны"


# Функции вычисления маршрута 

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Расстояние в метрах между двумя координатами (формула гаверсинусов).
    Точность достаточна для расстояний в пределах города.
    """
    R = 6_371_000  # Радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_waypoints(
    start: Coordinate, end: Coordinate, num_midpoints: int = 8
) -> List[Coordinate]:
    """
    Генерирует детерминированные waypoints для fallback-маршрута.

    Использует seed на основе координат, чтобы один и тот же маршрут
    всегда давал одинаковый результат (без случайного шума).
    """
    import random

    # Детерминированный seed на основе координат
    seed = hash((round(start.lat, 4), round(start.lng, 4),
                 round(end.lat, 4), round(end.lng, 4)))
    random.seed(seed)

    waypoints = []

    # Вычисляем общее расстояние для определения масштаба отклонений
    total_distance = haversine(start.lat, start.lng, end.lat, end.lng)

    # Отклонение пропорционально расстоянию (макс 200м для длинных маршрутов)
    max_deviation_deg = min(0.002, total_distance / 111000 / 10)

    for i in range(num_midpoints + 2):
        t = i / (num_midpoints + 1)
        base_lat = start.lat + (end.lat - start.lat) * t
        base_lng = start.lng + (end.lng - start.lng) * t

        # Добавляем детерминированное отклонение для средних точек
        if 0 < i < num_midpoints + 1:
            # Плавное отклонение: максимум в середине маршрута
            deviation_factor = math.sin(t * math.pi)  # 0 на концах, 1 в середине
            lat_offset = random.uniform(-max_deviation_deg, max_deviation_deg) * deviation_factor
            lng_offset = random.uniform(-max_deviation_deg, max_deviation_deg) * deviation_factor
            base_lat += lat_offset
            base_lng += lng_offset

        waypoints.append(Coordinate(lat=base_lat, lng=base_lng))

    return waypoints


def should_avoid_zone(zone: Zone, mode: str) -> bool:
    """Нужно ли избегать эту зону для данного режима?"""
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    return zone.danger_level > params.danger_threshold


def adjust_endpoints(
    start: Coordinate, end: Coordinate, mode: str, zones: List[Zone]
) -> tuple:
    """
    Проверяет, находятся ли start/end внутри опасных зон.
    Если да — сдвигает точку к ближайшей безопасной границе зоны.

    Returns:
        (adjusted_start, adjusted_end, warnings) — список предупреждений
    """
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    warnings = []

    lat_per_m = 1.0 / 111_000
    lng_per_m = 1.0 / (111_000 * math.cos(math.radians(50.41)))

    for label, point in [("start", start), ("end", end)]:
        for zone in zones:
            if not should_avoid_zone(zone, mode):
                continue

            dist = haversine(point.lat, point.lng, zone.lat, zone.lng)
            if dist < zone.radius:
                # Точка внутри опасной зоны — сдвигаем к границе
                dlat = point.lat - zone.lat
                dlng = point.lng - zone.lng

                # Если точка в центре — сдвигаем в сторону ближайшей безопасной точки
                if abs(dlat) < 1e-9 and abs(dlng) < 1e-9:
                    dlat = 0.001
                    dlng = 0.0

                vec_len = math.sqrt(dlat ** 2 + dlng ** 2)
                new_lat = zone.lat + (zone.radius + params.buffer_m) * lat_per_m * (dlat / vec_len)
                new_lng = zone.lng + (zone.radius + params.buffer_m) * lng_per_m * (dlng / vec_len)

                if label == "start":
                    start = Coordinate(lat=new_lat, lng=new_lng)
                else:
                    end = Coordinate(lat=new_lat, lng=new_lng)

                warnings.append(
                    f"⚠️ {label.capitalize()}-точка была в опасной зоне «{zone.name}» "
                    f"(уровень {zone.danger_level}/10) и сдвинута к безопасной границе."
                )

    return start, end, warnings


def adjust_route(
    waypoints: List[Coordinate], mode: str, zones: List[Zone]
) -> List[Coordinate]:
    """
    Корректирует маршрут, обходя опасные зоны.

    Алгоритм:
    1. Для каждой точки проверить расстояние до каждой зоны
    2. Если точка внутри опасной зоны — накопить вектор смещения
    3. Применить суммарное смещение (учитывает множественные зоны)
    4. Итерировать несколько раз для учёта пересечений зон
    5. Для туристов — притягивать к центру города
    6. Сглаживание маршрута в зависимости от режима
    """
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    adjusted = [Coordinate(lat=p.lat, lng=p.lng) for p in waypoints]

    # Приблизительная конвертация: 1 градус широты ≈ 111км
    # Для Семея: 1 градус долготы ≈ 111 * cos(50.4°) ≈ 70.7 км
    lat_per_m = 1.0 / 111_000
    lng_per_m = 1.0 / (111_000 * math.cos(math.radians(50.41)))

    # Фаза 1: Обход опасных зон (с накоплением смещений)
    for iteration in range(params.iterations):
        for i, point in enumerate(adjusted):
            # Пропускаем start и end точки
            if i == 0 or i == len(adjusted) - 1:
                continue

            # НАКОПИТЕЛЬНЫЕ смещения от всех зон
            total_dlat = 0.0
            total_dlng = 0.0
            zone_count = 0

            for zone in zones:
                if not should_avoid_zone(zone, mode):
                    continue

                dist = haversine(point.lat, point.lng, zone.lat, zone.lng)
                if dist >= zone.radius + params.buffer_m:
                    continue  # Точка вне зоны

                # Вектор от центра зоны к точке
                dlat = point.lat - zone.lat
                dlng = point.lng - zone.lng

                # Если точка точно в центре зоны — сдвинуть перпендикулярно
                if abs(dlat) < 1e-9 and abs(dlng) < 1e-9:
                    dlat = 0.001  # Примерно 100м на широте Семея
                    dlng = 0.0

                # Масштабировать вектор до radius + buffer
                vec_len = math.sqrt(dlat ** 2 + dlng ** 2)

                target_lat = zone.lat + (zone.radius + params.buffer_m) * lat_per_m * (dlat / vec_len)
                target_lng = zone.lng + (zone.radius + params.buffer_m) * lng_per_m * (dlng / vec_len)

                # НАКАПЛИВАЕМ смещение от каждой зоны
                total_dlat += (target_lat - point.lat)
                total_dlng += (target_lng - point.lng)
                zone_count += 1

            # Применяем суммарное смещение если точка в опасных зонах
            if zone_count > 0:
                # Среднее смещение от всех зон
                avg_dlat = total_dlat / zone_count
                avg_dlng = total_dlng / zone_count
                adjusted[i] = Coordinate(
                    lat=point.lat + avg_dlat,
                    lng=point.lng + avg_dlng
                )

    # Фаза 2: Притягивание к центру для туристов
    if params.prefer_central:
        for i in range(1, len(adjusted) - 1):  # Пропускаем start и end
            point = adjusted[i]
            dist_to_center = haversine(point.lat, point.lng, CITY_CENTER.lat, CITY_CENTER.lng)

            # Если точка далеко от центра, слегка притянуть
            if dist_to_center > CENTRAL_RADIUS_M:
                # Направление к центру
                dlat = CITY_CENTER.lat - point.lat
                dlng = CITY_CENTER.lng - point.lng
                vec_len = math.sqrt(dlat ** 2 + dlng ** 2)

                # Сдвиг на 20% расстояния к центру
                shift_factor = 0.2
                new_lat = point.lat + dlat * shift_factor
                new_lng = point.lng + dlng * shift_factor

                adjusted[i] = Coordinate(lat=new_lat, lng=new_lng)

    # Фаза 3: Сглаживание маршрута (moving average)
    if params.smooth_factor > 0:
        smoothed = adjusted.copy()
        for i in range(2, len(adjusted) - 2):  # Пропускаем start, end и соседние
            prev_point = adjusted[i - 1]
            curr_point = adjusted[i]
            next_point = adjusted[i + 1]

            # Среднее значение с учётом smooth_factor
            factor = params.smooth_factor
            new_lat = curr_point.lat + factor * ((prev_point.lat + next_point.lat) / 2 - curr_point.lat)
            new_lng = curr_point.lng + factor * ((prev_point.lng + next_point.lng) / 2 - curr_point.lng)

            smoothed[i] = Coordinate(lat=new_lat, lng=new_lng)

        adjusted = smoothed

    return adjusted


def calculate_danger_score(
    route: List[Coordinate], zones: List[Zone]
) -> float:
    """
    Средний уровень опасности вдоль маршрута (0-10).

    Для каждой точки: сумма влияния зон.
    Влияние зоны = danger_level * max(0, 1 - distance/radius)
    """
    if not route:
        return 0.0

    total = 0.0
    for point in route:
        point_danger = 0.0
        for zone in zones:
            dist = haversine(point.lat, point.lng, zone.lat, zone.lng)
            if dist < zone.radius:
                influence = 1.0 - (dist / zone.radius)
                point_danger += zone.danger_level * influence
        total += min(point_danger, 10.0)  # Cap per-point at 10

    score = total / len(route)
    return round(min(score, 10.0), 2)


def calculate_danger_score_with_ml(
    route: List[Coordinate],
    zones: List[Zone],
    predictor,
    hour: int = 12,
    day: int = 0,
) -> float:
    """
    Вычисляет danger_score используя комбинацию статических зон и ML предсказаний.

    Args:
        route: Список координат маршрута
        zones: Список зон опасности
        predictor: DangerPredictor instance
        hour: Час суток (0-23)
        day: День недели (0-6)

    Returns:
        Комбинированный danger score (0-10)

    Формула: combined = 0.4 * zones_score + 0.6 * ml_score
    - zones_score: основан на близости к статическим зонам
    - ml_score: основан на ML предсказаниях для каждой точки
    """
    if not route:
        return 0.0

    # 1. Zones score (статический метод)
    zones_score = calculate_danger_score(route, zones)

    # 2. ML predictions для каждой точки маршрута
    ml_predictions = []
    for point in route:
        try:
            result = predictor.predict_for_coordinates(
                lat=point.lat,
                lng=point.lng,
                hour=hour,
                day=day
            )
            ml_predictions.append(result.danger_level)
        except Exception:
            # Если ML предсказание не удалось, используем zones_score
            ml_predictions.append(zones_score)

    ml_score = sum(ml_predictions) / len(ml_predictions) if ml_predictions else 0.0

    # 3. Комбинация: 40% zones + 60% ML
    combined = 0.4 * zones_score + 0.6 * ml_score

    return round(min(combined, 10.0), 2)


def analyze_route_zones(
    route: List[Coordinate], zones: List[Zone]
) -> tuple[List[Dict], List[Dict]]:
    """
    Определяет какие зоны маршрут обошёл и рядом с какими прошёл.

    Returns:
        (zones_avoided, zones_nearby) — для промпта Claude
    """
    zones_avoided = []
    zones_nearby = []

    for zone in zones:
        min_dist = min(
            haversine(p.lat, p.lng, zone.lat, zone.lng)
            for p in route
        )

        entry = {
            "name": zone.name,
            "danger_level": zone.danger_level,
            "distance": int(min_dist),
            "description": zone.description,
        }

        if min_dist > zone.radius:
            zones_avoided.append(entry)
        elif min_dist < zone.radius * 1.5:
            zones_nearby.append(entry)

    return zones_avoided, zones_nearby


# Кэш для OSRM маршрутов (в памяти)
_osrm_cache: Dict[str, List[Coordinate]] = {}
OSRM_CACHE_SIZE = 100

# Маппинг режимов на OSRM профили
OSRM_PROFILES = {
    "car": "driving",
    "child": "walking",      # Ребёнок идёт пешком
    "tourist": "walking",    # Турист тоже пешком
}

async def get_osrm_route(start: Coordinate, end: Coordinate, mode: str = "car", max_retries: int = 2) -> Optional[List[Coordinate]]:
    """
    Получает реальный маршрут по дорогам через OSRM API с кэшированием и retry.

    Args:
        start: Начальная точка
        end: Конечная точка
        mode: Режим маршрута (car/child/tourist)
        max_retries: Максимальное количество попыток

    Returns:
        Список координат маршрута или None
    """
    # ВАЖНО: ключ кэша включает режим — иначе вернётся автомобильный маршрут для ребёнка
    profile = OSRM_PROFILES.get(mode, "driving")
    cache_key = f"{mode}:{round(start.lat, 4)},{round(start.lng, 4)}->{round(end.lat, 4)},{round(end.lng, 4)}"

    # Проверяем кэш
    if cache_key in _osrm_cache:
        print(f"[OSRM] Cache hit for {cache_key}")
        return _osrm_cache[cache_key]

    # Попытки получить маршрут с retry
    for attempt in range(max_retries):
        try:
            # OSRM demo server — используем профиль в зависимости от режима
            url = f"http://router.project-osrm.org/route/v1/{profile}/{start.lng},{start.lat};{end.lng},{end.lat}?overview=full&geometries=geojson"

            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("code") == "Ok" and data.get("routes"):
                        # Извлекаем координаты из GeoJSON geometry
                        geometry = data["routes"][0]["geometry"]["coordinates"]
                        # OSRM возвращает [lng, lat], нужно [lat, lng]
                        route_coords = [
                            Coordinate(lat=coord[1], lng=coord[0])
                            for coord in geometry
                        ]

                        # Сохраняем в кэш (с ограничением размера)
                        if len(_osrm_cache) >= OSRM_CACHE_SIZE:
                            # Удаляем самый старый элемент
                            oldest_key = next(iter(_osrm_cache))
                            del _osrm_cache[oldest_key]

                        _osrm_cache[cache_key] = route_coords
                        print(f"[OSRM] Route cached for {cache_key}")
                        return route_coords

        except httpx.TimeoutException:
            print(f"[OSRM] Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)  # Небольшая пауза перед retry
        except Exception as e:
            print(f"[OSRM error] Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.3)

    print(f"[OSRM] Failed after {max_retries} attempts")
    return None


def _point_in_polygon(lat: float, lng: float, polygon: List[Dict[str, float]]) -> bool:
    """
    Проверяет, находится ли точка внутри полигона (ray casting algorithm).

    Args:
        lat: широта точки
        lng: долгота точки
        polygon: список {"lat": ..., "lng": ...}

    Returns:
        True если точка внутри полигона
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi = polygon[i]["lat"]
        xi = polygon[i]["lng"]
        yj = polygon[j]["lat"]
        xj = polygon[j]["lng"]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def find_districts_on_route(
    route: List[Coordinate],
    districts: List[District]
) -> List[Dict]:
    """
    Определяет через какие районы проходит маршрут.
    """
    districts_on_route = []
    seen_ids = set()

    for point in route[::max(1, len(route) // 20)]:  # Проверяем каждые ~5% маршрута
        for district in districts:
            if district.id in seen_ids:
                continue

            # Используем ray casting для точной проверки попадания в полигон
            if _point_in_polygon(point.lat, point.lng, district.polygon):
                districts_on_route.append({
                    "id": district.id,
                    "name": district.name,
                    "danger_level": district.danger_level
                })
                seen_ids.add(district.id)

    return districts_on_route


# ===== API эндпоинты =====

@router.get("/zones", response_model=List[Zone])
async def get_zones_endpoint():
    """Возвращает список зон опасности для тепловой карты."""
    return get_zones()


@router.get("/districts", response_model=List[District])
async def get_districts_endpoint():
    """Возвращает список районов города с уровнями опасности."""
    return get_districts()


@router.patch("/districts/{district_id}")
async def update_district(
    district_id: str,
    request: DistrictUpdateRequest,
    req: Request
):
    """
    Обновляет уровень опасности района.

    Автоматически:
    1. Обновляет данные в памяти
    2. Сохраняет изменения в districts.json
    3. Обновляет ML predictor для синхронизации

    Args:
        district_id: ID района (например, 'center', 'cemposelok')
        request: новый danger_level и опциональное description

    Returns:
        Обновлённые данные района
    """
    districts = get_districts()
    target = None

    for district in districts:
        if district.id == district_id:
            target = district
            break

    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Район '{district_id}' не найден"
        )

    # Обновляем в памяти
    target.danger_level = request.danger_level
    if request.description is not None:
        target.description = request.description

    # Сохраняем в файл
    districts_path = Path(__file__).parent / "data" / "districts.json"
    try:
        with open(districts_path, "r", encoding="utf-8") as f:
            districts_data = json.load(f)

        # Находим и обновляем район в JSON
        for district_data in districts_data["districts"]:
            if district_data["id"] == district_id:
                district_data["danger_level"] = request.danger_level
                if request.description is not None:
                    district_data["description"] = request.description
                break

        with open(districts_path, "w", encoding="utf-8") as f:
            json.dump(districts_data, f, ensure_ascii=False, indent=2)

        print(f"[Districts] Обновлён район '{target.name}': danger_level={request.danger_level}")

    except Exception as e:
        print(f"[Districts] Ошибка сохранения: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения: {str(e)}"
        )

    # Обновляем ML predictor для синхронизации
    predictor = getattr(req.app.state, 'predictor', None)
    if predictor:
        predictor.set_districts([d.model_dump() for d in districts])
        print(f"[ML] Синхронизирован predictor с обновлёнными районами")

    return {
        "id": target.id,
        "name": target.name,
        "danger_level": target.danger_level,
        "description": target.description,
    }


@router.post("/districts/batch-update")
async def batch_update_districts(
    updates: Dict[str, int],
    req: Request
):
    """
    Массовое обновление уровней опасности районов.

    Args:
        updates: словарь {district_id: new_danger_level}

    Returns:
        Список обновлённых районов
    """
    districts = get_districts()
    updated = []

    for district_id, new_level in updates.items():
        if not (1 <= new_level <= 10):
            raise HTTPException(
                status_code=400,
                detail=f"Уровень опасности для '{district_id}' должен быть от 1 до 10"
            )

        for district in districts:
            if district.id == district_id:
                district.danger_level = new_level
                updated.append({
                    "id": district.id,
                    "name": district.name,
                    "danger_level": new_level,
                })
                break

    # Сохраняем в файл
    districts_path = Path(__file__).parent / "data" / "districts.json"
    try:
        with open(districts_path, "r", encoding="utf-8") as f:
            districts_data = json.load(f)

        for district_data in districts_data["districts"]:
            if district_data["id"] in updates:
                district_data["danger_level"] = updates[district_data["id"]]

        with open(districts_path, "w", encoding="utf-8") as f:
            json.dump(districts_data, f, ensure_ascii=False, indent=2)

        print(f"[Districts] Массовое обновление: {len(updated)} районов")

    except Exception as e:
        print(f"[Districts] Ошибка сохранения: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения: {str(e)}"
        )

    # Синхронизируем ML predictor
    predictor = getattr(req.app.state, 'predictor', None)
    if predictor:
        predictor.set_districts([d.model_dump() for d in districts])
        print(f"[ML] Синхронизирован predictor с обновлёнными районами")

    return {"updated": updated, "count": len(updated)}


@router.get("/search")
async def search_address(q: str = ""):
    """
    Прокси для поиска адресов через 2GIS API.
    Ключ API хранится только на бэкенде, не передаётся на фронтенд.
    Параметр: q (совпадает с фронтендом).
    """
    if not TWOGIS_API_KEY:
        raise HTTPException(status_code=503, detail="2GIS API ключ не настроен")

    if not q or len(q) < 2:
        return {"result": {"items": []}}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://catalog.api.2gis.com/3.0/items",
                params={
                    "q": q,
                    "fields": "items.point,items.adm_div,items.address_name,items.name,items.full_name,items.street",
                    "key": TWOGIS_API_KEY,
                    "page_size": 15
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"2GIS API error: {response.text[:200]}"
                )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="2GIS API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/geocode")
async def geocode(q: str = ""):
    """
    Геокодирование через Nominatim (OpenStreetMap).
    Бесплатный, быстрый, точный для Казахстана.
    Возвращает массив адресов с координатами.
    """
    if not q or len(q) < 2:
        return {"results": []}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            # Поиск в пределах Семея (viewbox bias)
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": q + ", Семей, Казахстан",
                    "format": "jsonv2",
                    "limit": 10,
                    "addressdetails": 1,
                    "countrycodes": "kz",
                    "viewbox": "80.12,50.36,80.35,50.49",  # [lon1,lat1,lon2,lat2] = [west,south,east,north]
                    "bounded": 0,  # Предпочитать Семей, но не ограничивать жёстко
                },
                headers={
                    "User-Agent": "SafeRouteAI/1.0 (hackathon project)",
                    "Accept-Language": "ru",
                }
            )

            if response.status_code != 200:
                print(f"[Nominatim] HTTP {response.status_code}: {response.text[:200]}")
                return {"results": []}

            data = response.json()

            results = []
            for item in data:
                lat = float(item.get("lat", 0))
                lng = float(item.get("lon", 0))

                # Фильтруем по границам Семея
                if not (50.35 <= lat <= 50.49 and 80.12 <= lng <= 80.35):
                    continue

                # Формируем читаемый адрес
                addr = item.get("address", {})
                display_name = item.get("display_name", "")
                street = addr.get("road", "")
                house = addr.get("house_number", "")
                suburb = addr.get("suburb", "")

                if street:
                    short_address = street
                    if house:
                        short_address += f", {house}"
                    if suburb:
                        short_address += f", {suburb}"
                else:
                    # Берём первую часть display_name
                    parts = display_name.split(",")
                    short_address = ", ".join(parts[:3]).strip()

                results.append({
                    "name": item.get("name", "") or short_address,
                    "address": short_address,
                    "lat": lat,
                    "lng": lng,
                    "type": item.get("type", ""),
                    "importance": item.get("importance", 0),
                })

            # Сортируем по важности
            results.sort(key=lambda x: x.get("importance", 0), reverse=True)

            return {"results": results}

    except httpx.TimeoutException:
        print("[Nominatim] Timeout")
        return {"results": []}
    except Exception as e:
        print(f"[Nominatim] Error: {e}")
        return {"results": []}


@router.get("/geocode/reverse")
async def geocode_reverse(lat: float, lng: float):
    """
    Обратное геокодирование через Nominatim.
    Возвращает адрес по координатам (для «Моё местоположение»).
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": lat,
                    "lon": lng,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "zoom": 18,
                },
                headers={
                    "User-Agent": "SafeRouteAI/1.0 (hackathon project)",
                    "Accept-Language": "ru",
                }
            )

            if response.status_code != 200:
                return {"name": "Моё местоположение", "address": "", "lat": lat, "lng": lng}

            data = response.json()
            addr = data.get("address", {})
            street = addr.get("road", "")
            house = addr.get("house_number", "")
            display_name = data.get("display_name", "")

            if street:
                short_address = street
                if house:
                    short_address += f", {house}"
            else:
                parts = display_name.split(",")
                short_address = ", ".join(parts[:3]).strip()

            return {
                "name": data.get("name", "") or short_address or "Моё местоположение",
                "address": short_address,
                "lat": float(data.get("lat", lat)),
                "lng": float(data.get("lon", lng)),
            }

    except Exception as e:
        print(f"[Nominatim reverse] Error: {e}")
        return {"name": "Моё местоположение", "address": "", "lat": lat, "lng": lng}


def detect_dangerous_districts_on_route(
    route: List[Coordinate],
    districts: List[District],
    mode: str
) -> List[Dict]:
    """
    Определяет через какие ОПАСНЫЕ районы проходит маршрут.

    Returns:
        Список опасных районов с процентом маршрута через них
    """
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    dangerous_districts = []

    # Проверяем каждую 20-ю точку маршрута (5% покрытия)
    sample_indices = list(range(0, len(route), max(1, len(route) // 20)))

    for district in districts:
        # Пропускаем безопасные районы
        if district.danger_level <= params.danger_threshold:
            continue

        # Считаем сколько точек маршрута попадает в этот район
        points_in_district = 0
        for idx in sample_indices:
            point = route[idx]
            if _point_in_polygon(point.lat, point.lng, district.polygon):
                points_in_district += 1

        if points_in_district > 0:
            percentage = (points_in_district / len(sample_indices)) * 100
            dangerous_districts.append({
                "id": district.id,
                "name": district.name,
                "danger_level": district.danger_level,
                "percentage_of_route": round(percentage, 1)
            })

    return dangerous_districts


@router.post("/route", response_model=RouteResponse)
async def compute_route(request: RouteRequest, req: Request):
    """
    Строит безопасный маршрут между start и end.

    1. Получает реальный маршрут по дорогам через OSRM
    2. Корректирует маршрут для обхода опасных зон
    3. Вычисляет danger_score (с ML если доступна)
    4. Определяет районы на маршруте
    5. Получает объяснение от AI
    """
    zones = get_zones()
    districts = get_districts()
    predictor = getattr(req.app.state, 'predictor', None)

    # 0. Проверяем и корректируем start/end если они внутри опасных зон
    start, end, warnings = adjust_endpoints(
        request.start, request.end, request.mode.value, zones
    )

    # 1. Получаем реальный маршрут по дорогам через OSRM (с учётом режима)
    osrm_route = await get_osrm_route(start, end, request.mode.value)

    if osrm_route and len(osrm_route) > 5:
        # КРИТИЧНО: корректируем OSRM маршрут для обхода опасных зон
        adjusted = adjust_route(osrm_route, request.mode.value, zones)
        print(f"[Route] OSRM: {len(adjusted)} точек по реальным дорогам (скорректировано)")
    else:
        # Fallback: генерируем waypoints + корректировка (используем скорректированные start/end)
        waypoints = generate_waypoints(start, end, num_midpoints=8)
        adjusted = adjust_route(waypoints, request.mode.value, zones)
        warnings.append("⚠️ OSRM недоступен — маршрут построен по прямой линии. Может быть менее точным.")
        print(f"[Route] Fallback: {len(adjusted)} точек (без OSRM)")

    # 2. Danger score (с ML если доступна)
    if predictor:
        from datetime import datetime
        now = datetime.now()
        hour = request.hour if request.hour is not None else now.hour
        day = request.day if request.day is not None else now.weekday()

        score = calculate_danger_score_with_ml(
            adjusted, zones, predictor, hour=hour, day=day
        )
    else:
        score = calculate_danger_score(adjusted, zones)

    # 3. Анализ зон для AI
    avoided, nearby = analyze_route_zones(adjusted, zones)

    # 4. Определяем районы через которые проходит маршрут
    districts_on_route = find_districts_on_route(adjusted, districts)

    # 5. Проверяем опасные районы на маршруте
    dangerous_districts = detect_dangerous_districts_on_route(
        adjusted, districts, request.mode.value
    )

    # Если маршрут проходит через опасные районы (>30% маршрута) - предупреждаем
    route_buildable = True
    if dangerous_districts:
        total_dangerous_percentage = sum(d["percentage_of_route"] for d in dangerous_districts)
        if total_dangerous_percentage > 30:
            route_buildable = False
            district_names = ", ".join([d["name"] for d in dangerous_districts])
            warnings.append(
                f"🚨 НЕВОЗМОЖНО построить безопасный маршрут! "
                f"Маршрут проходит через опасные районы: {district_names} "
                f"({total_dangerous_percentage:.0f}% пути). "
                f"Рекомендуем изменить точки старта/финиша."
            )
        else:
            for district in dangerous_districts:
                warnings.append(
                    f"⚠️ Маршрут проходит через район «{district['name']}» "
                    f"(опасность {district['danger_level']}/10, {district['percentage_of_route']}% пути). "
                    f"Будьте осторожны!"
                )

    # 6. AI объяснение с реальными данными
    from ai import generate_route_explanation
    explanation = await generate_route_explanation(
        route=[{"lat": p.lat, "lng": p.lng} for p in adjusted],
        mode=request.mode.value,
        zones_avoided=avoided,
        zones_nearby=nearby,
        danger_score=score,
        districts_on_route=districts_on_route,
    )

    return RouteResponse(
        route=adjusted,
        danger_score=score,
        ai_explanation=explanation,
        warnings=warnings,
        route_buildable=route_buildable,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_danger(request: PredictRequest, req: Request):
    """
    Предсказывает уровень опасности по координатам используя ML модель.

    Автоматически определяет признаки района:
    - Тип района (по удалённости от центра)
    - Освещение (по времени суток)
    - Плотность населения и камер
    - Исторические инциденты
    """
    predictor = getattr(req.app.state, 'predictor', None)
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="ML модель не загружена. Проверьте наличие danger_model.pkl"
        )

    result = predictor.predict_for_coordinates(
        request.lat, request.lng, request.hour, request.day
    )

    return result.to_dict()


@router.get("/ml/info")
async def ml_info(req: Request):
    """
    Возвращает информацию о ML модели и важность признаков.

    Используется для демонстрации ML жюри и frontend визуализации.
    """
    predictor = getattr(req.app.state, 'predictor', None)
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="ML модель не загружена"
        )

    return {
        "model_info": predictor.get_model_info(),
        "feature_importance": predictor.get_feature_importance(),
        "feature_descriptions": predictor.get_feature_descriptions(),
        "statistics": predictor.get_statistics(),
    }


class HeatmapRequest(BaseModel):
    """Запрос для генерации heatmap на основе ML."""
    hour: int = Field(12, ge=0, le=23, description="Час суток (0-23)")
    day: int = Field(0, ge=0, le=6, description="День недели (0-6)")
    bounds: Optional[Dict[str, float]] = Field(
        None,
        description="Границы карты: {north, south, east, west}"
    )
    grid_size: int = Field(20, ge=10, le=50, description="Размер сетки (10-50)")


class HeatmapCell(BaseModel):
    """Ячейка heatmap с ML предсказанием (полигон)."""
    polygon: List[Dict[str, float]]  # 4 угла квадрата: [{lat, lng}, ...]
    danger_level: int
    confidence: float
    risk_category: str


@router.post("/ml/heatmap", response_model=List[HeatmapCell])
async def generate_heatmap(request: HeatmapRequest, req: Request):
    """
    Генерирует heatmap на основе ML предсказаний.

    Создаёт сетку полигонов (квадратов) и для каждого получает ML предсказание
    уровня опасности. Полигоны покрывают всю область без промежутков.
    """
    predictor = getattr(req.app.state, 'predictor', None)
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="ML модель не загружена"
        )

    # Границы Семея (полный охват города)
    if request.bounds:
        north = request.bounds.get('north', 50.49)
        south = request.bounds.get('south', 50.36)
        east = request.bounds.get('east', 80.35)
        west = request.bounds.get('west', 80.12)
    else:
        north, south = 50.49, 50.36
        east, west = 80.35, 80.12

    # Размер каждой ячейки
    lat_step = (north - south) / request.grid_size
    lng_step = (east - west) / request.grid_size

    cells = []
    for i in range(request.grid_size):
        for j in range(request.grid_size):
            # Координаты центра ячейки
            center_lat = south + (i + 0.5) * lat_step
            center_lng = west + (j + 0.5) * lng_step

            # Получаем ML предсказание для центра
            try:
                result = predictor.predict_for_coordinates(
                    lat=center_lat,
                    lng=center_lng,
                    hour=request.hour,
                    day=request.day
                )

                # Вычисляем 4 угла квадрата
                sw_lat = south + i * lat_step
                sw_lng = west + j * lng_step
                ne_lat = sw_lat + lat_step
                ne_lng = sw_lng + lng_step

                polygon = [
                    {"lat": sw_lat, "lng": sw_lng},  # Юго-запад
                    {"lat": ne_lat, "lng": sw_lng},  # Северо-запад
                    {"lat": ne_lat, "lng": ne_lng},  # Северо-восток
                    {"lat": sw_lat, "lng": ne_lng},  # Юго-восток
                ]

                cells.append(HeatmapCell(
                    polygon=polygon,
                    danger_level=result.danger_level,
                    confidence=result.confidence,
                    risk_category=result.risk_category
                ))
            except Exception:
                continue

    return cells
