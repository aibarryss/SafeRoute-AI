"""
AI генерация объяснений для безопасных маршрутов.
Использует OpenRouter API с контекстом реальных районов Семея.
"""
import os
import httpx
from typing import List, Dict

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
TWOGIS_API_KEY = os.getenv("TWOGIS_API_KEY")
if not TWOGIS_API_KEY:
    print("[WARNING] TWOGIS_API_KEY не задан в .env — geocoding и карта работать не будут")

# Реальные данные о районах Семея для контекста AI
SEMEY_DISTRICTS_INFO = """
Районы Семея (Семипалатинска), Казахстан:

ЦЕНТРАЛЬНЫЕ РАЙОНЫ (безопасные, уровень 2-3):
- Центр (правый берег) — исторический центр, площадь Абая, Центральный парк
- Карагайлы — новый современный район, комплекс "Арена"
- Юность — уютный жилой район
- Татарский край — исторический район XIX века
- Алаш-кала — исторический район

ЖИЛЫЕ РАЙОНЫ (умеренно безопасные, уровень 4-5):
- Океан — жилой район
- Энергетик — микрорайон с многоэтажками
- Новостройка — жилой массив
- Степной — микрорайон
- Жоламан (левый берег) — частный сектор
- Затон — район у речного порта

ПРОМЫШЛЕННЫЕ РАЙОНЫ (менее безопасные, уровень 6-7):
- Цемпоселок — район цементного завода
- Мясокомбинат — промышленная зона
- Обувная фабрика — промзона
- Силикатный — заводской поселок
- Восточный — частный сектор на окраине
"""


async def reverse_geocode(lat: float, lng: float) -> str:
    """
    Получает название места по координатам через 2GIS API.
    Fallback на Nominatim если 2GIS недоступен.
    """
    # Пробуем 2GIS (только если ключ задан)
    if TWOGIS_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://catalog.api.2gis.com/3.0/items/geocode",
                    params={
                        "lat": lat,
                        "lon": lng,
                        "fields": "items.point,items.adm_div,items.address_name,items.name,items.street",
                        "key": TWOGIS_API_KEY
                    },
                    headers={"User-Agent": "SafeRoute-AI/1.0"},
                    timeout=5.0
                )
            if response.status_code == 200:
                data = response.json()
                items = data.get("result", {}).get("items", [])
                if items:
                    item = items[0]
                    # Берём название или адрес
                    name = (
                        item.get("name", "")
                        or item.get("address_name", "")
                        or item.get("street", "")
                    )
                    if name:
                        return name
        except Exception as e:
            print(f"[2GIS Geocoding error] {e}")

    # Fallback: Nominatim
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": lat,
                    "lon": lng,
                    "format": "json",
                    "accept-language": "ru"
                },
                headers={"User-Agent": "SafeRoute-AI/1.0"},
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                addr = data.get("address", {})
                name = (
                    addr.get("road", "")
                    or addr.get("suburb", "")
                    or addr.get("neighbourhood", "")
                    or addr.get("city_district", "")
                    or data.get("display_name", "").split(",")[0]
                )
                return name if name else data.get("display_name", "").split(",")[0]
    except Exception as e:
        print(f"[Nominatim fallback error] {e}")

    return "Неизвестное место"


def build_prompt(
    start_name: str,
    end_name: str,
    mode: str,
    avoided_zones: List[Dict],
    nearby_zones: List[Dict],
    danger_score: float,
    districts_on_route: List[Dict],
) -> str:
    """
    Строит промпт с реальными названиями мест.
    """
    mode_ru = {
        "car": "на автомобиле",
        "child": "с ребёнком",
        "tourist": "для туриста"
    }.get(mode, mode)

    mode_advice = {
        "car": "Дай совет для водителя: где безопасно припарковаться, какие улицы лучше освещены.",
        "child": "Дай совет для родителя: обрати внимание на тротуары, пешеходные переходы, людные места.",
        "tourist": "Дай совет для туриста: упомяни интересные места по пути, безопасные фото-точки.",
    }.get(mode, "")

    # Районы на маршруте
    districts_text = ""
    if districts_on_route:
        names = [d["name"] for d in districts_on_route]
        districts_text = f"\nМаршрут проходит через районы: {', '.join(names)}."

    # Обойдённые зоны
    avoided_text = ""
    if avoided_zones:
        lines = [f"  - {z['name']} (опасность: {z['danger_level']}/10)" for z in avoided_zones[:4]]
        avoided_text = f"\nОбойдённые опасные районы:\n" + "\n".join(lines)

    # Близкие зоны
    nearby_text = ""
    if nearby_zones:
        lines = [f"  - {z['name']} (опасность: {z['danger_level']}/10, {z['distance']}м от маршрута)" for z in nearby_zones[:3]]
        nearby_text = f"\nРайоны рядом с маршрутом:\n" + "\n".join(lines)

    # Оценка безопасности
    if danger_score <= 3:
        safety = "НИЗКИЙ — маршрут очень безопасный"
    elif danger_score <= 6:
        safety = "СРЕДНИЙ — маршрут умеренно безопасный"
    else:
        safety = "ПОВЫШЕННЫЙ — будьте внимательны"

    return f"""Ты — ассистент по безопасности городской среды в городе Семей (Семипалатинск), Казахстан.

Объясни пользователю почему построенный маршрут безопасен.

МАРШРУТ:
Откуда: {start_name}
Куда: {end_name}
Режим: {mode_ru}
Оценка безопасности: {danger_score}/10 ({safety})
{districts_text}{avoided_text}{nearby_text}

СПРАВОЧНИК РАЙОНОВ СЕМЕЯ:
{SEMEY_DISTRICTS_INFO}

ПРАВИЛА:
1. Пиши кратко: 3-4 предложения, простым русским языком
2. Используй ТОЛЬКО реальные названия районов из справочника выше
3. НЕ выдумывай улицы, магазины или места — только районы
4. НЕ используй markdown, эмодзи, списки — только обычный текст
5. {mode_advice}
6. Если оценка <= 3 — подчеркни что маршрут очень безопасен
7. Если оценка >= 7 — предупреди о повышенной осторожности

Твой ответ (3-4 предложения, без заголовков):"""


async def generate_route_explanation(
    route: List[Dict[str, float]],
    mode: str,
    zones_avoided: List[Dict],
    zones_nearby: List[Dict],
    danger_score: float = 5.0,
    districts_on_route: List[Dict] = None,
) -> str:
    """
    Генерирует объяснение маршрута через AI.
    При ошибке возвращает fallback.
    """
    if not OPENROUTER_API_KEY:
        return _fallback_explanation(mode, zones_avoided, danger_score, districts_on_route)

    try:
        # Получаем реальные названия начальной и конечной точек
        start_name = await reverse_geocode(route[0]["lat"], route[0]["lng"])
        end_name = await reverse_geocode(route[-1]["lat"], route[-1]["lng"])

        prompt = build_prompt(
            start_name, end_name, mode,
            zones_avoided, zones_nearby, danger_score,
            districts_on_route or [],
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://saferoute.ai",
                    "X-Title": "SafeRoute AI",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 250,
                    "temperature": 0.5
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[OpenRouter error] {response.status_code}: {response.text[:200]}")

    except Exception as e:
        print(f"[AI error] {e}")

    return _fallback_explanation(mode, zones_avoided, danger_score, districts_on_route)


def _fallback_explanation(
    mode: str,
    zones_avoided: List[Dict],
    danger_score: float,
    districts_on_route: List[Dict] = None,
) -> str:
    """Fallback объяснение без AI."""
    mode_ru = {"car": "на автомобиле", "child": "с ребёнком", "tourist": "для туриста"}.get(mode, mode)

    if danger_score <= 3:
        safety = "Этот маршрут очень безопасен"
    elif danger_score <= 6:
        safety = "Этот маршрут умеренно безопасен"
    else:
        safety = "Этот маршрут требует повышенной осторожности"

    parts = [f"{safety}. Маршрут {mode_ru} построен с учётом уровня безопасности районов Семея."]

    if districts_on_route:
        names = ", ".join(d["name"] for d in districts_on_route[:3])
        parts.append(f"Маршрут проходит через: {names}.")

    if zones_avoided:
        names = ", ".join(z["name"] for z in zones_avoided[:3])
        parts.append(f"Обойдены опасные районы: {names}.")

    parts.append("(AI-объяснение временно недоступно)")
    return " ".join(parts)
