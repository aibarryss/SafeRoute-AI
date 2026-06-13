# SafeRoute AI 🗺️

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![2GIS](https://img.shields.io/badge/2GIS-MapGL-00A98F?logo=2gis&logoColor=white)](https://2gis.ru/)
[![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?logo=vercel&logoColor=white)](https://vercel.com/)
[![Railway](https://img.shields.io/badge/Railway-Backend-0B0D0E?logo=railway&logoColor=white)](https://railway.app/)

**SafeRoute AI** — интеллектуальный планировщик безопасных маршрутов для города Семей, Казахстан. Разработано для хакатона (Track 3 — City Safety & Social Services).

Пользователь выбирает режим (🚗 машина / 👶 ребёнок / 🧳 турист), вводит начальную и конечную точки — система строит маршрут на карте 2GIS, избегая опасных зон, и объясняет на русском языке, почему маршрут безопасен.

## 🚀 Live Demo

**🌐 [saferoute-a1.vercel.app](https://saferoute-a1.vercel.app)**

## ✨ Возможности

- **Три режима маршрутизации** — оптимизация для автомобиля, ребёнка или туриста
- **ML-модель предсказания опасности** — Gradient Boosting классификатор на основе 8 признаков района
- **Интерактивная карта 2GIS** — визуализация маршрута и опасных зон в реальном времени
- **AI-объяснения** — нейросеть генерирует понятные объяснения безопасности маршрута (на русском)
- **Динамическое управление районами** — редактирование уровней опасности через UI
- **Геокодирование** — поиск адресов через 2GIS API (основной) и Nominatim (fallback)

## 🛠️ Технологический стек

| Слой | Технологии |
|------|------------|
| **Frontend** | HTML5, JavaScript, 2GIS MapGL SDK, CSS3 |
| **Backend** | Python 3.11+, FastAPI, Uvicorn, httpx |
| **Machine Learning** | scikit-learn, NumPy, Pandas, joblib |
| **AI** | OpenRouter API (Claude 3.5 Sonnet) |
| **Карты** | 2GIS MapGL, OSRM Routing |
| **Деплой** | Vercel (frontend), Railway (backend) |

## 📊 Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vercel)                        │
│   HTML + JavaScript + 2GIS MapGL                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP API
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (Railway)                          │
│   FastAPI + Uvicorn                                             │
│   ├── ML Model (Gradient Boosting)                              │
│   ├── District Management                                       │
│   ├── Route Computation                                         │
│   └── AI Explanation Generator                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │ 2GIS API│         │OpenRouter│         │   OSRM   │
   │Geocoder │         │   API    │         │ Routing  │
   └─────────┘         └──────────┘         └──────────┘
```

## 🚦 Быстрый старт

### Требования

- Python 3.11+
- API ключи: OpenRouter, 2GIS

### Установка

**1. Клонируйте репозиторий:**

```bash
git clone https://github.com/aibarryss/SafeRoute-AI.git
cd SafeRoute-AI
```

**2. Установите зависимости:**

```bash
pip install -r requirements.txt
```

**3. Создайте файл `.env` в корне проекта:**

```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ
OPENROUTER_MODEL=openrouter/owl-alpha
TWOGIS_API_KEY=ваш-2gis-ключ
ALLOWED_ORIGINS=http://localhost:5500,http://localhost:8080
```

> **Получение ключей:**
> - OpenRouter: [openrouter.ai/keys](https://openrouter.ai/keys)
> - 2GIS: [dev.2gis.com](https://dev.2gis.com/)

**4. Запустите backend:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**5. Откройте frontend:**

Откройте `frontend/index.html` в браузере или запустите через Live Server (VS Code) на порту 5500.

## 📡 API Endpoints

### Основные

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/api/config` | Публичная конфигурация (2GIS ключ для frontend) |
| `POST` | `/api/route` | Построение безопасного маршрута |
| `POST` | `/api/predict` | ML-предсказание опасности для точки |
| `GET` | `/api/ml/info` | Информация о ML-модели и важность признаков |

### Районы

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/api/districts` | Все районы с полигонами и уровнями опасности |
| `PATCH` | `/api/districts/{id}` | Обновить уровень опасности района |
| `POST` | `/api/districts/batch-update` | Массовое обновление районов |

### Геокодирование

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/api/search?q=...` | Прямой геокодинг (2GIS) |
| `GET` | `/api/geocode?q=...` | Поиск через Nominatim (fallback) |
| `GET` | `/api/geocode/reverse?lat=...&lng=...` | Обратный геокодинг |

### Пример запроса

```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "start": {"lat": 50.4111, "lng": 80.2275},
    "end": {"lat": 50.4200, "lng": 80.2400},
    "mode": "car"
  }'
```

## 🎯 Режимы маршрутизации

| Режим | Порог опасности | Буфер | Итерации | Особенности |
|-------|----------------|-------|----------|-------------|
| 🚗 **Машина** | > 6 | 80м | 3 | Может объехать дальше |
| 👶 **Ребёнок** | > 3 | 120м | 5 | Максимальная безопасность, плавный маршрут |
| 🧳 **Турист** | > 6 | 100м | 4 | Предпочитает центральные улицы |

## 🤖 ML-модель

**Алгоритм:** Gradient Boosting Classifier (scikit-learn)

**Признаки:**
- `hour_of_day` (0-23) — время суток
- `day_of_week` (0-6) — день недели
- `district_type` (0-4) — тип района (центр, жилой, промышленный, рынок, парк)
- `lighting` (0-1) — уровень освещённости
- `cctv_density` (0-1) — плотность камер видеонаблюдения
- `population_density` (0-1) — плотность населения
- `historical_incidents` (0-100) — количество инцидентов
- `road_type` (0-3) — тип дороги (магистраль, основная, второстепенная, пешеходная)

**Выход:** `danger_level` (1-10), `confidence` (0-1), `risk_category` (low/medium/high/critical)

**Accuracy:** 0.6283 (F1-macro: 0.5847)

**Гибридный скоринг:** 40% статические зоны + 60% ML-предсказания

## 📁 Структура проекта

```
SafeRoute-AI/
├── backend/
│   ├── main.py                    # FastAPI точка входа
│   ├── routes.py                  # API endpoints + логика маршрутов
│   ├── ai.py                      # OpenRouter AI интеграция
│   ├── data/
│   │   ├── districts.json         # Районы Семея с полигонами
│   │   └── semey_zones.json       # Legacy круговые зоны
│   └── ml/
│       ├── danger_predictor.py    # Production ML inference
│       ├── train_model.py         # Обучение модели
│       ├── data_generator.py      # Генерация синтетических данных
│       └── models/
│           ├── danger_model.pkl   # Обученная модель
│           └── model_metrics.json # Метрики модели
├── frontend/
│   ├── index.html                 # HTML структура
│   ├── app.js                     # Логика карты, геокодинг, маршруты
│   └── style.css                  # Стили UI
├── requirements.txt               # Python зависимости
├── Procfile                       # Railway деплой
├── railway.json                   # Railway конфигурация
├── vercel.json                    # Vercel конфигурация
├── .railwayignore                 # Исключения для Railway
└── .env.example                   # Пример переменных окружения
```

## 🚀 Деплой

### Backend (Railway)

1. Создайте проект на [railway.app](https://railway.app)
2. Подключите GitHub-репозиторий
3. Railway автоматически обнаружит `Procfile` и `railway.json`
4. Добавьте переменные окружения в Railway Dashboard:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL`
   - `TWOGIS_API_KEY`
   - `ALLOWED_ORIGINS` (URL вашего Vercel-деплоя)

### Frontend (Vercel)

1. Создайте проект на [vercel.com](https://vercel.com)
2. Подключите GitHub-репозиторий
3. Настройки:
   - **Framework Preset:** Other
   - **Build Command:** (оставьте пустым)
   - **Output Directory:** `frontend`
4. Vercel автоматически использует `vercel.json`

## 🔄 Переобучение ML-модели

```bash
cd backend/ml
python train_model.py
```

Модель будет сохранена в `backend/ml/models/danger_model.pkl`.

## 📝 Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OPENROUTER_API_KEY` | API ключ OpenRouter | — |
| `OPENROUTER_MODEL` | Модель для AI-объяснений | `openrouter/owl-alpha` |
| `TWOGIS_API_KEY` | API ключ 2GIS | — |
| `ALLOWED_ORIGINS` | Разрешённые CORS домены | `http://localhost:*` |

## ⚠️ Известные ограничения

1. **Синтетические данные** — районы и зоны опасности сгенерированы искусственно, но структурированы реалистично. Архитектура готова к замене на реальные источники (данные полиции, жалобы граждан, городские камеры).

2. **OSRM как роутинг-бэкенд** — маршруты получаются с демо-сервера OSRM. При недоступности система fallback-ит на прямые waypoints.

3. **Cold start на Railway** — первый запрос может занимать 5-10 секунд из-за загрузки ML-модели.

## 📄 Лицензия

MIT

## 👥 Авторы

Разработано для хакатона (Track 3 — City Safety & Social Services).

## 🔗 Ссылки

- **Live Demo:** [saferoute-a1.vercel.app](https://saferoute-a1.vercel.app)
- **Backend API:** [saferoute-ai-production-f06b.up.railway.app](https://saferoute-ai-production-f06b.up.railway.app)
- **GitHub:** [github.com/aibarryss/SafeRoute-AI](https://github.com/aibarryss/SafeRoute-AI)
