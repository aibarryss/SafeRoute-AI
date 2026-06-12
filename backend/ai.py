"""
Модуль интеграции с OpenRouter AI API.
Генерирует объяснения маршрутов на русском языке.
"""

import os
from typing import List, Dict, Optional
import openai

# Модель через OpenRouter (можно сменить на любую доступную)
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/owl-alpha")
MAX_TOKENS = 500

# Клиент инициализируется лениво
_client: Optional[openai.AsyncOpenAI] = None


def _get_client() -> openai.AsyncOpenAI:
    """Ленивая инициализация клиента OpenRouter API."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY не задан")
        _client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/SafeRoute-AI",
                "X-Title": "SafeRoute AI",
            },
        )
    return _client


def _build_prompt(
    route: List[Dict[str, float]],
    mode: str,
    zones_avoided: List[Dict],
    zones_nearby: List[Dict],
) -> str:
    """Строит промпт на основе данных маршрута."""

    mode_labels = {
        "car": "на машине",
        "child": "с ребёнком",
        "tourist": "для туриста",
    }
    mode_russian = mode_labels.get(mode, mode)

    # Формируем описание маршрута
    route_desc = " → ".join(
        f"({p['lat']:.4f}, {p['lng']:.4f})" for p in route
    )

    # Формируем список обойдённых зон
    avoided_desc = "\n".join(
        f"  - {z['name']} (уровень опасности: {z['danger_level']}/10, "
        f"расстояние: {z['distance']}м, {z.get('description', '')})"
        for z in zones_avoided
    ) if zones_avoided else "  (нет опасных зон на пути)"

    nearby_desc = "\n".join(
        f"  - {z['name']} (уровень опасности: {z['danger_level']}/10, "
        f"расстояние: {z['distance']}м, {z.get('description', '')})"
        for z in zones_nearby
    ) if zones_nearby else "  (нет)"

    return f"""Ты — эксперт по безопасности маршрутов в городе Семей, Казахстан.
Система SafeRoute AI построила безопасный маршрут для пользователя.

Режим: {mode_russian}
Маршрут (координаты): {route_desc}

Опасные зоны, которые маршрут успешно обошёл:
{avoided_desc}

Опасные зоны рядом с маршрутом (маршрут прошёл поблизости):
{nearby_desc}

Напиши краткое объяснение (3-5 предложений) на русском языке, почему этот маршрут
безопасен. Упомяни конкретные районы, которые были обойдены, и дай практический
совет. Используй простой и дружелюбный тон. НЕ используй markdown, только текст."""


async def generate_route_explanation(
    route: List[Dict[str, float]],
    mode: str,
    zones_avoided: List[Dict],
    zones_nearby: List[Dict],
) -> str:
    """
    Генерирует объяснение маршрута с помощью OpenRouter API.
    При ошибке возвращает fallback-сообщение.
    """
    prompt = _build_prompt(route, mode, zones_avoided, zones_nearby)

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    except ValueError:
        # OPENROUTER_API_KEY не задан
        return _fallback_explanation(mode, zones_avoided)
    except (openai.APITimeoutError, openai.APIError) as e:
        print(f"[ai.py] Ошибка OpenRouter API: {e}")
        return _fallback_explanation(mode, zones_avoided)
    except Exception as e:
        # Любая другая ошибка (несовместимость версий, сеть и т.д.)
        print(f"[ai.py] Непредвиденная ошибка: {e}")
        return _fallback_explanation(mode, zones_avoided)


def _fallback_explanation(
    mode: str, zones_avoided: List[Dict]
) -> str:
    """Fallback объяснение если AI API недоступен."""
    mode_labels = {"car": "на машине", "child": "с ребёнком", "tourist": "для туриста"}
    mode_russian = mode_labels.get(mode, mode)
    avoided_names = ", ".join(z["name"] for z in zones_avoided[:3])

    base = f"Этот маршрут {mode_russian} построен с учётом безопасности."
    if avoided_names:
        base += f" Маршрут обходит районы: {avoided_names}."
    base += " (AI-объяснение временно недоступно)"
    return base
