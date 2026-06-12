"""
API маршруты и логика вычисления безопасных маршрутов.
"""

import math
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from fastapi import APIRouter, Request, HTTPException


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
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    return zone.danger_level > params.danger_threshold


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
    5. Для туристов — притягивать к центру города
    6. Сглаживание маршрута в зависимости от режима
    """
    params = MODE_PARAMS.get(mode, MODE_PARAMS["car"])
    adjusted = [Coordinate(lat=p.lat, lng=p.lng) for p in waypoints]

    # Фаза 1: Обход опасных зон
    for _ in range(params.iterations):
        for i, point in enumerate(adjusted):
            # Пропускаем start и end точки
            if i == 0 or i == len(adjusted) - 1:
                continue

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
                # Приблизительная конвертация: 1 градус широты ≈ 111км
                # Для Семея: 1 градус долготы ≈ 111 * cos(50.4°) ≈ 70.7 км
                lat_per_m = 1.0 / 111_000
                lng_per_m = 1.0 / (111_000 * math.cos(math.radians(zone.lat)))

                target_lat = zone.lat + (zone.radius + params.buffer_m) * lat_per_m * (dlat / vec_len)
                target_lng = zone.lng + (zone.radius + params.buffer_m) * lng_per_m * (dlng / vec_len)

                adjusted[i] = Coordinate(lat=target_lat, lng=target_lng)

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


# ===== API эндпоинты =====

@router.get("/zones", response_model=List[Zone])
async def get_zones_endpoint():
    """Возвращает список зон опасности для тепловой карты."""
    return get_zones()


@router.post("/route", response_model=RouteResponse)
async def compute_route(request: RouteRequest, req: Request):
    """
    Строит безопасный маршрут между start и end.

    1. Генерирует промежуточные точки
    2. Корректирует маршрут для обхода опасных зон
    3. Вычисляет danger_score (с ML если доступна)
    4. Получает объяснение от AI
    """
    zones = get_zones()
    predictor = getattr(req.app.state, 'predictor', None)

    # 1. Начальные точки
    waypoints = generate_waypoints(request.start, request.end, num_midpoints=6)

    # 2. Корректировка маршрута
    adjusted = adjust_route(waypoints, request.mode.value, zones)

    # 3. Danger score (с ML если доступна)
    if predictor:
        score = calculate_danger_score_with_ml(
            adjusted, zones, predictor, hour=12, day=0
        )
    else:
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
