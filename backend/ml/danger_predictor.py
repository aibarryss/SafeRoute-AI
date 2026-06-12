"""
Инференс-класс для предсказания уровня опасности района.

Production-ready предиктор с:
- Валидацией входных данных
- Кэшированием предсказаний
- Мониторингом латентности
- Batch predictions
- Graceful degradation

Загружает обученную модель и предоставляет методы для:
- predict(): предсказание danger_level по признакам
- predict_proba(): вероятности для каждого уровня
- predict_batch(): пакетное предсказание для маршрута
- predict_for_coordinates(): автоматическое определение признаков по координатам
- get_feature_importance(): важность признаков

Использование:
    from ml.danger_predictor import DangerPredictor

    predictor = DangerPredictor()
    result = predictor.predict({
        "hour_of_day": 22,
        "day_of_week": 5,
        "district_type": 2,
        "lighting": 0.3,
        "cctv_density": 0.1,
        "population_density": 0.2,
        "historical_incidents": 30,
        "road_type": 2,
    })
    print(result)
    # {
    #     "danger_level": 7,
    #     "confidence": 0.85,
    #     "probabilities": {...},
    #     "latency_ms": 2.3,
    #     "risk_category": "high"
    # }
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass, asdict
import numpy as np
import joblib

from ml.data_generator import get_feature_names, FEATURE_DESCRIPTIONS


# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Пути к модели
ML_DIR = Path(__file__).parent
MODEL_DIR = ML_DIR / "models"
MODEL_PATH = MODEL_DIR / "danger_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


@dataclass
class PredictionResult:
    """
    Структурированный результат предсказания.

    Attributes:
        danger_level: предсказанный уровень опасности (1-10)
        confidence: уверенность модели (0-1)
        probabilities: вероятности для каждого уровня
        latency_ms: время выполнения в миллисекундах
        risk_category: категория риска (low/medium/high/critical)
        features_used: использованные признаки
    """
    danger_level: int
    confidence: float
    probabilities: Dict[int, float]
    latency_ms: float
    risk_category: str
    features_used: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return asdict(self)


class DangerPredictor:
    """
    Production-ready предиктор уровня опасности района.

    Особенности:
    - Валидация входных данных
    - Кэширование предсказаний (LRU cache)
    - Мониторинг латентности
    - Batch predictions для маршрутов
    - Graceful degradation при ошибках

    Attributes:
        model: обученный GradientBoostingClassifier
        feature_names: список названий признаков
        feature_importance: словарь важности признаков
        metrics: метрики обученной модели
        cache_enabled: включено ли кэширование
        cache_hits: количество попаданий в кэш
        total_predictions: общее количество предсказаний
    """

    # Категории риска
    RISK_CATEGORIES = {
        (1, 3): "low",
        (4, 6): "medium",
        (7, 8): "high",
        (9, 10): "critical",
    }

    # Допустимые диапазоны признаков
    FEATURE_RANGES = {
        "hour_of_day": (0, 23),
        "day_of_week": (0, 6),
        "district_type": (0, 4),
        "lighting": (0.0, 1.0),
        "cctv_density": (0.0, 1.0),
        "population_density": (0.0, 1.0),
        "historical_incidents": (0, 100),
        "road_type": (0, 3),
    }

    def __init__(
        self,
        model_path: str = None,
        cache_enabled: bool = True,
        cache_size: int = 1000,
    ):
        """
        Инициализация и загрузка модели.

        Args:
            model_path: путь к файлу модели (по умолчанию models/danger_model.pkl)
            cache_enabled: включить ли LRU кэш для предсказаний
            cache_size: максимальный размер кэша

        Raises:
            FileNotFoundError: если модель не найдена
        """
        path = Path(model_path) if model_path else MODEL_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {path}\n"
                f"Сначала обучите модель: python -m backend.ml.train_model"
            )

        logger.info(f"Загрузка модели: {path}")
        self.model = joblib.load(path)
        self.feature_names = get_feature_names()
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size

        # Статистика
        self.cache_hits = 0
        self.total_predictions = 0
        self.total_latency_ms = 0.0

        # Загрузка метрик
        self._load_metrics()

        # Инициализация кэша
        if cache_enabled:
            self._init_cache(cache_size)

        logger.info(f"Модель загружена: {len(self.feature_names)} признаков, "
                   f"accuracy={self.metrics.get('accuracy', 'N/A')}")

    def _load_metrics(self):
        """Загружает метрики модели из JSON."""
        self.metrics = {}
        self.feature_importance = {}

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
                self.feature_importance = self.metrics.get("feature_importance", {})
            logger.info(f"Метрики загружены: {METRICS_PATH}")
        else:
            logger.warning(f"Метрики не найдены: {METRICS_PATH}")

    def _init_cache(self, cache_size: int):
        """Инициализирует LRU кэш для предсказаний."""
        # Кэшируем внутреннюю функцию предсказания
        @lru_cache(maxsize=cache_size)
        def cached_predict(features_key: tuple) -> dict:
            features = dict(features_key)
            return self._predict_internal(features)

        self._cached_predict = cached_predict

    def _validate_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Валидирует входные признаки.

        Args:
            features: словарь признаков

        Returns:
            (is_valid, errors) — валидны ли признаки и список ошибок
        """
        errors = []

        # Проверка наличия всех признаков
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            errors.append(f"Отсутствуют признаки: {missing}")
            return False, errors

        # Проверка диапазонов
        for feature_name, (min_val, max_val) in self.FEATURE_RANGES.items():
            if feature_name in features:
                value = features[feature_name]
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"{feature_name}={value} вне диапазона [{min_val}, {max_val}]"
                    )

        return len(errors) == 0, errors

    def _predict_internal(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Внутренняя функция предсказания (без кэширования)."""
        X = np.array([[features[f] for f in self.feature_names]])

        # Предсказание
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Уверенность
        classes = self.model.classes_
        pred_idx = list(classes).index(prediction)
        confidence = float(probabilities[pred_idx])

        # Вероятности для каждого уровня
        prob_dict = {}
        for cls, prob in zip(classes, probabilities):
            prob_dict[int(cls)] = round(float(prob), 4)

        return {
            "danger_level": int(prediction),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def _get_risk_category(self, danger_level: int) -> str:
        """Определяет категорию риска по уровню опасности."""
        for (min_level, max_level), category in self.RISK_CATEGORIES.items():
            if min_level <= danger_level <= max_level:
                return category
        return "unknown"

    def predict(
        self,
        features: Dict[str, Any],
        validate: bool = True,
    ) -> PredictionResult:
        """
        Предсказывает уровень опасности района.

        Args:
            features: словарь с 8 признаками:
                - hour_of_day (0-23): время суток
                - day_of_week (0-6): день недели
                - district_type (0-4): тип района
                - lighting (0-1): уровень освещённости
                - cctv_density (0-1): плотность камер
                - population_density (0-1): плотность населения
                - historical_incidents (0-100): количество инцидентов
                - road_type (0-3): тип дороги
            validate: валидировать ли входные данные

        Returns:
            PredictionResult с danger_level, confidence, probabilities, latency

        Raises:
            ValueError: если валидация включена и признаки невалидны
        """
        start_time = time.perf_counter()
        self.total_predictions += 1

        # Валидация
        if validate:
            is_valid, errors = self._validate_features(features)
            if not is_valid:
                raise ValueError(f"Невалидные признаки: {'; '.join(errors)}")

        # Предсказание (с кэшированием или без)
        if self.cache_enabled:
            # Создаём hashable key для кэша
            features_key = tuple(sorted(features.items()))
            result_dict = self._cached_predict(features_key)

            # Проверка cache hit
            cache_info = self._cached_predict.cache_info()
            if cache_info.hits > self.cache_hits:
                self.cache_hits += 1
                logger.debug(f"Cache hit: {features}")
        else:
            result_dict = self._predict_internal(features)

        # Вычисление латентности
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.total_latency_ms += latency_ms

        # Категория риска
        risk_category = self._get_risk_category(result_dict["danger_level"])

        # Создание результата
        result = PredictionResult(
            danger_level=result_dict["danger_level"],
            confidence=result_dict["confidence"],
            probabilities=result_dict["probabilities"],
            latency_ms=round(latency_ms, 3),
            risk_category=risk_category,
            features_used=features,
        )

        logger.debug(f"Предсказание: danger={result.danger_level}, "
                    f"confidence={result.confidence:.2%}, "
                    f"latency={result.latency_ms:.2f}ms")

        return result

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        validate: bool = True,
    ) -> List[PredictionResult]:
        """
        Пакетное предсказание для нескольких точек (например, маршрут).

        Args:
            features_list: список словарей признаков
            validate: валидировать ли входные данные

        Returns:
            Список PredictionResult для каждой точки
        """
        start_time = time.perf_counter()

        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.predict(features, validate=validate)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка предсказания #{i}: {e}")
                # Graceful degradation — возвращаем среднее значение
                results.append(PredictionResult(
                    danger_level=5,
                    confidence=0.0,
                    probabilities={},
                    latency_ms=0.0,
                    risk_category="unknown",
                    features_used=features,
                ))

        total_latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Batch prediction: {len(results)} точек за {total_latency:.2f}ms")

        return results

    def predict_for_coordinates(
        self,
        lat: float,
        lng: float,
        hour: int = 12,
        day: int = 0,
    ) -> PredictionResult:
        """
        Предсказывает опасность для конкретной точки на карте.

        Автоматически определяет признаки района по координатам:
        - Тип района (по удалённости от центра)
        - Освещение (по времени суток)
        - Плотность населения и камер (по типу района)

        Args:
            lat: широта
            lng: долгота
            hour: время суток (0-23)
            day: день недели (0-6)

        Returns:
            PredictionResult с автоматически определёнными признаками
        """
        import math

        # Центр Семея
        CITY_CENTER_LAT = 50.4111
        CITY_CENTER_LNG = 80.2275

        # Расстояние от центра (приблизительно)
        dlat = lat - CITY_CENTER_LAT
        dlng = lng - CITY_CENTER_LNG
        dist_deg = math.sqrt(dlat ** 2 + dlng ** 2)
        dist_m = dist_deg * 111_000  # грубая конвертация

        # Определение типа района по расстоянию и направлению
        if dist_m < 1500:
            district_type = 0  # центр
            cctv = 0.7
            lighting_base = 0.8
            pop_density = 0.6
            incidents = 15
            road_type = 0
        elif dist_m < 3000:
            district_type = 1  # жилой
            cctv = 0.3
            lighting_base = 0.5
            pop_density = 0.5
            incidents = 8
            road_type = 1
        elif dlng > 0.02:
            district_type = 2  # промышленный (восток)
            cctv = 0.15
            lighting_base = 0.3
            pop_density = 0.2
            incidents = 25
            road_type = 2
        else:
            district_type = 1  # жилой (по умолчанию)
            cctv = 0.3
            lighting_base = 0.5
            pop_density = 0.5
            incidents = 10
            road_type = 2

        # Освещение зависит от времени
        if 6 <= hour <= 20:
            lighting = lighting_base
        else:
            lighting = lighting_base * 0.3

        features = {
            "hour_of_day": hour,
            "day_of_week": day,
            "district_type": district_type,
            "lighting": round(min(lighting, 1.0), 3),
            "cctv_density": round(cctv, 3),
            "population_density": round(pop_density, 3),
            "historical_incidents": incidents,
            "road_type": road_type,
        }

        result = self.predict(features, validate=True)
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Возвращает важность признаков модели.

        Returns:
            Словарь {feature_name: importance} отсортированный по убыванию
        """
        return dict(self.feature_importance)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.

        Returns:
            Словарь с метриками, параметрами и статистикой
        """
        avg_latency = (
            self.total_latency_ms / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        cache_hit_rate = (
            self.cache_hits / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "model_type": "GradientBoostingClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "statistics": {
                "total_predictions": self.total_predictions,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "avg_latency_ms": round(avg_latency, 3),
            },
            "cache_enabled": self.cache_enabled,
        }

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Возвращает описания признаков."""
        return dict(FEATURE_DESCRIPTIONS)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования предиктора.

        Returns:
            Словарь со статистикой предсказаний
        """
        avg_latency = (
            self.total_latency_ms / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(
                self.cache_hits / self.total_predictions
                if self.total_predictions > 0
                else 0.0,
                4
            ),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "avg_latency_ms": round(avg_latency, 3),
        }

    def reset_statistics(self):
        """Сбрасывает статистику предсказаний."""
        self.total_predictions = 0
        self.cache_hits = 0
        self.total_latency_ms = 0.0
        logger.info("Статистика предиктора сброшена")


if __name__ == "__main__":
    # Тестирование предиктора
    print("=" * 70)
    print("  SafeRoute AI - Тестирование DangerPredictor")
    print("=" * 70)

    # Загрузка предиктора
    try:
        predictor = DangerPredictor(cache_enabled=True)
        print(f"\n[OK] Предиктор загружен")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nЗапустите обучение модели:")
        print("  python -m backend.ml.train_model")
        exit(1)

    # Информация о модели
    info = predictor.get_model_info()
    print(f"\nМодель: {info['model_type']}")
    print(f"Accuracy: {info['metrics'].get('accuracy', 'N/A')}")
    print(f"F1-macro: {info['metrics'].get('f1_macro', 'N/A')}")
    print(f"Признаков: {info['n_features']}")

    # Тест 1: Предсказание по признакам
    print("\n" + "=" * 70)
    print("  Тест 1: Предсказание по признакам")
    print("=" * 70)

    test_features = {
        "hour_of_day": 22,
        "day_of_week": 5,
        "district_type": 2,  # промзона
        "lighting": 0.3,
        "cctv_density": 0.1,
        "population_density": 0.2,
        "historical_incidents": 30,
        "road_type": 2,
    }

    result = predictor.predict(test_features)
    print(f"\nВходные признаки:")
    for k, v in test_features.items():
        desc = FEATURE_DESCRIPTIONS.get(k, "")
        print(f"  {k:25s} = {v:6}  ({desc})")

    print(f"\nРезультат:")
    print(f"  Danger Level:  {result.danger_level}/10")
    print(f"  Confidence:    {result.confidence:.2%}")
    print(f"  Risk Category: {result.risk_category}")
    print(f"  Latency:       {result.latency_ms:.2f}ms")

    # Тест 2: Предсказание по координатам
    print("\n" + "=" * 70)
    print("  Тест 2: Предсказание по координатам")
    print("=" * 70)

    test_coords = [
        (50.4111, 80.2275, "Центр Семея"),
        (50.4200, 80.2500, "Восточная промзона"),
        (50.4050, 80.2100, "Жилой район"),
    ]

    for lat, lng, name in test_coords:
        result = predictor.predict_for_coordinates(lat, lng, hour=22, day=5)
        print(f"\n{name} ({lat}, {lng}):")
        print(f"  Danger Level:  {result.danger_level}/10")
        print(f"  Confidence:    {result.confidence:.2%}")
        print(f"  Risk Category: {result.risk_category}")

    # Тест 3: Batch prediction
    print("\n" + "=" * 70)
    print("  Тест 3: Batch prediction (маршрут из 5 точек)")
    print("=" * 70)

    batch_features = [
        {"hour_of_day": 12, "day_of_week": 2, "district_type": 0, "lighting": 0.9,
         "cctv_density": 0.8, "population_density": 0.7, "historical_incidents": 10, "road_type": 0},
        {"hour_of_day": 12, "day_of_week": 2, "district_type": 1, "lighting": 0.7,
         "cctv_density": 0.4, "population_density": 0.6, "historical_incidents": 8, "road_type": 1},
        {"hour_of_day": 12, "day_of_week": 2, "district_type": 1, "lighting": 0.6,
         "cctv_density": 0.3, "population_density": 0.5, "historical_incidents": 12, "road_type": 2},
        {"hour_of_day": 12, "day_of_week": 2, "district_type": 0, "lighting": 0.8,
         "cctv_density": 0.7, "population_density": 0.6, "historical_incidents": 9, "road_type": 1},
        {"hour_of_day": 12, "day_of_week": 2, "district_type": 0, "lighting": 0.9,
         "cctv_density": 0.8, "population_density": 0.7, "historical_incidents": 11, "road_type": 0},
    ]

    results = predictor.predict_batch(batch_features)
    for i, result in enumerate(results):
        print(f"  Точка {i+1}: danger={result.danger_level}, "
              f"confidence={result.confidence:.2%}, "
              f"risk={result.risk_category}")

    # Статистика
    print("\n" + "=" * 70)
    print("  Статистика предиктора")
    print("=" * 70)

    stats = predictor.get_statistics()
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Cache hits:        {stats['cache_hits']}")
    print(f"  Cache hit rate:    {stats['cache_hit_rate']:.2%}")
    print(f"  Avg latency:       {stats['avg_latency_ms']:.2f}ms")

    print("\n" + "=" * 70)
    print("  Тестирование завершено!")
    print("=" * 70)
