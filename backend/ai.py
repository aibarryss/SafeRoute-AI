"""
AI генерация объяснений для безопасных маршрутов.
Использует OpenRouter API с улучшенными промптами.
"""
import os
import httpx
from typing import List, Dict

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Реальные данные о районах Семея для контекста
SEMEY_DISTRICTS_INFO = """
Районы Семея (Семипалатинска), Казахстан:

ЦЕНТРАЛЬНЫЕ РАЙОНЫ (безопасные):
- Центр (правый берег) - исторический центр, площадь Абая, Центральный парк
- Карагайлы - новый современный район, комплекс "Арена"
- Юность - уютный жилой район
- Татарский край - исторический район XIX века
- Алаш-кала - исторический район

ЖИЛЫЕ РАЙОНЫ (умеренно безопасные):
- Океан - жилой район
- Энергетик - микрорайон с многоэтажками
- Новостройка - жилой массив
- Степной - микрорайон
- Жоламан (левый берег) - частный сектор
- Затон - район у речного порта

ПРОМЫШЛЕННЫЕ РАЙОНЫ (менее безопасные):
- Цемпоселок - район цементного завода
- Мясокомбинат - промышленная зона
- Обувная фабрика - промзона
- Силикатный - заводской поселок
- Восточный - частный сектор на окраине
"""


async def reverse_geocode(lat: float, lng: float) -> str:
    """
    Получает название места по координатам через Nominatim.
    """
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
                return data.get("display_name", f"Координаты {lat:.4f}, {lng:.4f}")
    except Exception as e:
        print(f"[Geocoding error] {e}")

    return f"Координаты {lat:.4f}, {lng:.4f}"


def build_improved_prompt(
    start_name: str,
    end_name: str,
    mode: str,
    avoided_zones: List[Dict],
    nearby_zones: List[Dict],
    danger_score: float
) -> str:
    """
    Строит улучшенный промпт с реальными названиями мест.
    """
    mode_ru = {
        "car": "на автомобиле",
        "child": "с ребёнком",
        "tourist": "для туриста"
    }.get(mode, mode)

    # Форматируем обойдённые зоны
    avoided_text = ""
    if avoided_zones:
        avoided_list = "\n".join([
            f"  - {z['name']} (уровень опасности: {z['danger_level']}/10)"
            for z in avoided_zones[:5]  # Максимум 5 зон
        ])
        avoided_text = f"\n\nОбойдённые опасные зоны:\n{avoided_list}"

    # Форматируем близкие зоны
    nearby_text = ""
    if nearby_zones:
        nearby_list = "\n".join([
            f"  - {z['name']} (уровень опасности: {z['danger_level']}/10, расстояние: {z['distance']}м)"
            for z in nearby_zones[:3]  # Максимум 3 зоны
        ])
        nearby_text = f"\n\nЗоны рядом с маршрутом:\n{nearby_list}"

    prompt = f"""Ты - эксперт по безопасности городской среды в Семее (Казахстан). Объясни пользователю почему этот маршрут безопасен.

МАРШРУТ:
Откуда: {start_name}
Куда: {end_name}
Режим: {mode_ru}
Общий уровень опасности маршрута: {danger_score}/10 (чем ниже, тем безопаснее)
{avoided_text}{nearby_text}

КОНТЕКСТ О СЕМЕЕ:
{SEMEY_DISTRICTS_INFO}

ИНСТРУКЦИИ:
1. Пиши кратко (3-5 предложений), простым языком
2. Используй ТОЛЬКО реальные названия районов из контекста выше
3. НЕ выдумывай несуществующие улицы или места
4. Объясни почему маршрут безопасен (обошли опасные зоны)
5. Дай 1 практический совет для режима "{mode_ru}"
6. Если danger_score <= 3, скажи что маршрут очень безопасен
7. Если danger_score 4-6, скажи что маршрут умеренно безопасен
8. Если danger_score >= 7, предупреди о повышенной осторожности

Пример хорошего ответа:
"Этот маршрут проходит через центр Семея и район Карагайлы — одни из самых безопасных районов города. Мы обошли промышленную зону Цемпоселок и Мясокомбинат. Для поездки с ребёнком это оптимальный путь с хорошим освещением и людными улицами."

Твой ответ (только текст объяснения, без заголовков):"""

    return prompt


async def generate_route_explanation(
    route: List[Dict[str, float]],
    mode: str,
    zones_avoided: List[Dict],
    zones_nearby: List[Dict],
    danger_score: float = 5.0
) -> str:
    """
    Генерирует объяснение маршрута через AI с улучшенным промптом.
    """
    if not OPENROUTER_API_KEY:
        return _fallback_explanation(mode, zones_avoided, danger_score)

    try:
        # Получаем названия начальной и конечной точек
        start_name = await reverse_geocode(route[0]["lat"], route[0]["lng"])
        end_name = await reverse_geocode(route[-1]["lat"], route[-1]["lng"])

        # Строим улучшенный промпт
        prompt = build_improved_prompt(
            start_name, end_name, mode,
            route, zones_avoided, zones_nearby, danger_score
        )

        # Запрос к OpenRouter
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
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                data = response.json()
                explanation = data["choices"][0]["message"]["content"]
                return explanation.strip()
            else:
                print(f"[OpenRouter error] Status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[AI generation error] {e}")

    return _fallback_explanation(mode, zones_avoided, danger_score)


def _fallback_explanation(mode: str, zones_avoided: List[Dict], danger_score: float) -> str:
    """
    Fallback объяснение если AI недоступен.
    """
    mode_ru = {
        "car": "на автомобиле",
        "child": "с ребёнком",
        "tourist": "для туриста"
    }.get(mode, mode)

    if danger_score <= 3:
        safety = "очень безопасен"
    elif danger_score <= 6:
        safety = "умеренно безопасен"
    else:
        safety = "требует повышенной осторожности"

    if zones_avoided:
        avoided_names = ", ".join([z["name"] for z in zones_avoided[:3]])
        return f"Этот маршрут {mode_ru} {safety} (уровень опасности: {danger_score}/10). Мы обошли опасные зоны: {avoided_names}. Будьте внимательны в пути!"
    else:
        return f"Этот маршрут {mode_ru} {safety} (уровень опасности: {danger_score}/10). Будьте внимательны в пути!"
