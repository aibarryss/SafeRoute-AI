"""
API маршруты и логика вычисления безопасных маршрутов.
"""

import math
from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum
from fastapi import APIRouter


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


class RouteResponse(BaseModel):
    """Ответ с построенным маршрутом."""
    route: List[Coordinate]
    danger_score: float = Field(..., ge=0, le=10)
    ai_explanation: str


class Zone(BaseModel):
    """Зона опасности."""
    id: str
    name: str
    danger_level: int = Field(..., ge=1, le=10)
    lat: float
    lng: float
    radius: float = Field(..., gt=0, description="Радиус зоны в метрах")
    description: str = ""


# Router и глобальное состояни

router = APIRouter(prefix="/api", tags=["routes"])

# Глобальное хранилище зон — загружается в main.py через set_zones()
_zones: List[Zone] = []


def set_zones(zones: List[Zone]) -> None:
    """Устанавливает данные зон (вызывается при старте сервера)."""
    global _zones
    _zones = zones


def get_zones() -> List[Zone]:
    """Возвращает загруженные зоны."""
    return _zones


# Константы

DANGER_THRESHOLDS: Dict[str, int] = {
    "car": 6,       # Избегает danger > 6 (аварии, угоны)
    "child": 3,     # Только зелёные зоны (danger <= 3)
    "tourist": 6,   # Избегает красные зоны (danger > 6)
}


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
    start: Coordinate, end: Coordinate, num_midpoints: int = 6
) -> List[Coordinate]:
    """
    Генерирует num_midpoints + 2 точек (включая start и end)
    путём линейной интерполяции.
    """
    waypoints = []
    for i in range(num_midpoints + 2):
        t = i / (num_midpoints + 1)
        lat = start.lat + (end.lat - start.lat) * t
        lng = start.lng + (end.lng - start.lng) * t
        waypoints.append(Coordinate(lat=lat, lng=lng))
    return waypoints


def should_avoid_zone(zone: Zone, mode: str) -> bool:
    """Нужно ли избегать эту зону для данного режима?"""
    return zone.danger_level > DANGER_THRESHOLDS.get(mode, 6)


def adjust_route(
    waypoints: List[Coordinate], mode: str, zones: List[Zone]
) -> List[Coordinate]:
    """
    Корректирует маршрут, обходя опасные зоны.

    Алгоритм:
    1. Для каждой точки проверить расстояние до каждой зоны
    2. Если точка внутри опасной зоны — сдвинуть наружу
    3. Сдвиг: от центра зоны через точку, на radius + buffer
    4. Итерировать несколько раз для учёта пересечений зон
    """
    adjusted = [Coordinate(lat=p.lat, lng=p.lng) for p in waypoints]
    buffer_m = 50  # Запас за границей зоны в метрах
    iterations = 3  # Количество проходов для сглаживания

    for _ in range(iterations):
        for i, point in enumerate(adjusted):
            for zone in zones:
                if not should_avoid_zone(zone, mode):
                    continue

                dist = haversine(point.lat, point.lng, zone.lat, zone.lng)
                if dist >= zone.radius + buffer_m:
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
                # Приблизительная конвертация: 1 градус широты ≈ 111км
                # Для Семея: 1 градус долготы ≈ 111 * cos(50.4°) ≈ 70.7 км
                lat_per_m = 1.0 / 111_000
                lng_per_m = 1.0 / (111_000 * math.cos(math.radians(zone.lat)))

                target_lat = zone.lat + (zone.radius + buffer_m) * lat_per_m * (dlat / vec_len)
                target_lng = zone.lng + (zone.radius + buffer_m) * lng_per_m * (dlng / vec_len)

                adjusted[i] = Coordinate(lat=target_lat, lng=target_lng)

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


# ===== API эндпоинты =====

@router.get("/zones", response_model=List[Zone])
async def get_zones_endpoint():
    """Возвращает список зон опасности для тепловой карты."""
    return get_zones()


@router.post("/route", response_model=RouteResponse)
async def compute_route(request: RouteRequest):
    """
    Строит безопасный маршрут между start и end.

    1. Генерирует промежуточные точки
    2. Корректирует маршрут для обхода опасных зон
    3. Вычисляет danger_score
    4. Получает объяснение от Claude API
    """
    zones = get_zones()

    # 1. Начальные точки
    waypoints = generate_waypoints(request.start, request.end, num_midpoints=6)

    # 2. Корректировка маршрута
    adjusted = adjust_route(waypoints, request.mode.value, zones)

    # 3. Danger score
    score = calculate_danger_score(adjusted, zones)

    # 4. Анализ зон для AI
    avoided, nearby = analyze_route_zones(adjusted, zones)

    # 5. AI объяснение
    from ai import generate_route_explanation
    explanation = await generate_route_explanation(
        route=[{"lat": p.lat, "lng": p.lng} for p in adjusted],
        mode=request.mode.value,
        zones_avoided=avoided,
        zones_nearby=nearby,
    )

    return RouteResponse(
        route=adjusted,
        danger_score=score,
        ai_explanation=explanation,
    )
