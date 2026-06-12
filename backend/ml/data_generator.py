"""
Генерация синтетических данных для обучения модели предсказания опасности.

Признаки (features):
- hour_of_day (0-23): время суток
- day_of_week (0-6): день недели (0=пн, 6=вс)
- district_type (0-4): тип района (центр, жилой, промышленный, рынок, парк)
- lighting (0-1): уровень освещённости
- cctv_density (0-1): плотность камер видеонаблюдения
- population_density (0-1): плотность населения
- historical_incidents (0-100): количество исторических инцидентов
- road_type (0-3): тип дороги (магистраль, основная, второстепенная, пешеходная)

Целевая переменная:
- danger_level (1-10): уровень опасности района
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


# Типы районов Семея
DISTRICT_TYPES = {
    0: "центр",
    1: "жилой",
    2: "промышленный",
    3: "рынок",
    4: "парк",
}

# Типы дорог
ROAD_TYPES = {
    0: "магистраль",
    1: "основная",
    2: "второстепенная",
    3: "пешеходная",
}

# Описание признаков для документации
FEATURE_DESCRIPTIONS = {
    "hour_of_day": "Время суток (0-23)",
    "day_of_week": "День недели (0=Пн, 6=Вс)",
    "district_type": "Тип района: центр(0), жилой(1), промзона(2), рынок(3), парк(4)",
    "lighting": "Уровень освещённости (0=темно, 1=хорошо освещено)",
    "cctv_density": "Плотность камер видеонаблюдения (0=нет, 1=много)",
    "population_density": "Плотность населения в районе (0=пустынно, 1=очень людно)",
    "historical_incidents": "Количество зарегистрированных инцидентов за последний год",
    "road_type": "Тип дороги: магистраль(0), основная(1), второстепенная(2), пешеходная(3)",
}


def balance_classes(
    df: pd.DataFrame,
    target_col: str = "danger_level",
    strategy: str = "oversample",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Балансировка классов для целевой переменной.

    Стратегии:
    - oversample: дублирование миноритарных классов
    - undersample: удаление избыточных примеров из мажоритарных классов
    - hybrid: комбинация обоих методов

    Args:
        df: DataFrame с данными
        target_col: название целевой колонки
        strategy: стратегия балансировки
        random_state: seed для воспроизводимости

    Returns:
        Сбалансированный DataFrame
    """
    # Подсчёт примеров в каждом классе
    class_counts = df[target_col].value_counts()
    median_count = int(class_counts.median())

    balanced_dfs = []

    for danger_level in sorted(df[target_col].unique()):
        class_df = df[df[target_col] == danger_level]
        current_count = len(class_df)

        if strategy == "oversample":
            # Дублируем миноритарные классы до медианного значения
            if current_count < median_count:
                n_add = median_count - current_count
                additional = class_df.sample(n=n_add, replace=True, random_state=random_state)
                class_df = pd.concat([class_df, additional], ignore_index=True)
            else:
                # Оставляем как есть или слегка undersample
                if current_count > median_count * 1.5:
                    class_df = class_df.sample(n=median_count, random_state=random_state)

        elif strategy == "undersample":
            # Удаляем избыточные примеры из мажоритарных классов
            if current_count > median_count:
                class_df = class_df.sample(n=median_count, random_state=random_state)

        elif strategy == "hybrid":
            # Комбинация: oversample миноритарные, undersample мажоритарные
            if current_count < median_count * 0.8:
                n_add = int(median_count * 0.8) - current_count
                additional = class_df.sample(n=n_add, replace=True, random_state=random_state)
                class_df = pd.concat([class_df, additional], ignore_index=True)
            elif current_count > median_count * 1.2:
                class_df = class_df.sample(n=int(median_count * 1.2), random_state=random_state)

        balanced_dfs.append(class_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Перемешивание
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df


def generate_synthetic_data(
    n_samples: int = 5000,
    random_state: int = 42,
    save_path: str = None,
    balance_strategy: str = "oversample",
) -> pd.DataFrame:
    """
    Генерирует синтетический датасет для обучения модели опасности.

    Логика генерации основана на реальных городских паттернах Семея:

    **Временные паттерны:**
    - Ночью (0-5) опасность выше в жилых и промышленных районах
    - Днём (10-18) опасность ниже в центре, выше на рынках
    - Вечером (18-22) умеренная опасность

    **Географические паттерны:**
    - Центр: хорошее освещение, камеры, низкая опасность днём
    - Жилые районы: средняя опасность, зависит от времени
    - Промзоны: высокая опасность, плохое освещение, мало камер
    - Рынки: высокая опасность днём (карманные кражи), низкая ночью
    - Парки: безопасны днём, опасны ночью

    **Инфраструктурные паттерны:**
    - Хорошее освещение снижает опасность на 2-3 уровня
    - Камеры видеонаблюдения снижают опасность на 1-2 уровня
    - Высокая плотность населения: хорошо днём (свидетели), плохо ночью

    **Балансировка классов:**
    После генерации применяется балансировка для равномерного
    распределения danger_level (1-10). Стратегия по умолчанию: oversample.

    Args:
        n_samples: количество примеров для генерации
        random_state: seed для воспроизводимости
        save_path: путь для сохранения CSV (опционально)
        balance_strategy: стратегия балансировки (oversample/undersample/hybrid/none)

    Returns:
        DataFrame с признаками и целевой переменной danger_level
    """
    rng = np.random.default_rng(random_state)

    # === Генерация признаков ===

    # Время суток — равномерное распределение с акцентом на часы пик
    hour_weights = np.ones(24)
    hour_weights[7:10] = 2.0   # утренний пик
    hour_weights[17:20] = 2.0  # вечерний пик
    hour_weights[0:5] = 0.5    # ночью меньше данных
    hour_weights /= hour_weights.sum()
    hour_of_day = rng.choice(24, size=n_samples, p=hour_weights)

    # День недели — равномерное распределение
    day_of_week = rng.integers(0, 7, size=n_samples)

    # Тип района — реалистичное распределение районов Семея
    district_weights = [0.15, 0.40, 0.15, 0.10, 0.20]  # центр, жилой, пром, рынок, парк
    district_type = rng.choice(5, size=n_samples, p=district_weights)

    # Освещение — зависит от времени суток и типа района
    lighting = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        # Базовое освещение от времени
        if 6 <= hour_of_day[i] <= 20:
            base_light = 0.7
        else:
            base_light = 0.2

        # Модификатор от типа района
        district_mod = {0: 0.2, 1: 0.1, 2: -0.1, 3: 0.0, 4: -0.2}
        light = base_light + district_mod.get(district_type[i], 0)
        lighting[i] = np.clip(light + rng.normal(0, 0.1), 0, 1)

    # Плотность камер — выше в центре, ниже в промзонах
    cctv_base = {0: 0.7, 1: 0.3, 2: 0.15, 3: 0.4, 4: 0.1}
    cctv_density = np.array([
        np.clip(cctv_base.get(d, 0.3) + rng.normal(0, 0.15), 0, 1)
        for d in district_type
    ])

    # Плотность населения — зависит от времени и района
    population_density = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        # Базовая плотность от района
        pop_base = {0: 0.6, 1: 0.5, 2: 0.2, 3: 0.8, 4: 0.3}
        base = pop_base.get(district_type[i], 0.4)

        # Модификатор от времени суток
        if 0 <= hour_of_day[i] <= 5:
            time_mod = -0.3
        elif 7 <= hour_of_day[i] <= 9:
            time_mod = 0.1
        elif 17 <= hour_of_day[i] <= 19:
            time_mod = 0.2
        else:
            time_mod = 0.0

        population_density[i] = np.clip(base + time_mod + rng.normal(0, 0.1), 0, 1)

    # Исторические инциденты — зависят от типа района
    incident_base = {0: 15, 1: 8, 2: 25, 3: 20, 4: 3}
    historical_incidents = np.array([
        max(0, int(incident_base.get(d, 10) + rng.normal(0, 8)))
        for d in district_type
    ])

    # Тип дороги
    road_type = rng.integers(0, 4, size=n_samples)

    # === Вычисление целевой переменной (danger_level) ===
    # Реалистичная формула, отражающая реальные зависимости

    danger_raw = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        d = 5.0  # базовый уровень

        # Время суток: ночь опаснее
        h = hour_of_day[i]
        if 0 <= h <= 4:
            d += 2.0
        elif 5 <= h <= 7:
            d += 0.5
        elif 20 <= h <= 23:
            d += 1.0
        elif 10 <= h <= 17:
            d -= 1.0

        # День недели: выходные немного опаснее ночью
        if day_of_week[i] >= 5 and (h >= 22 or h <= 3):
            d += 1.0

        # Тип района
        district_danger = {0: -0.5, 1: 0.0, 2: 2.5, 3: 1.5, 4: -0.5}
        d += district_danger.get(district_type[i], 0)

        # Парки опасны ночью
        if district_type[i] == 4 and (h >= 21 or h <= 5):
            d += 2.5

        # Освещение снижает опасность
        d -= lighting[i] * 2.5

        # Камеры снижают опасность
        d -= cctv_density[i] * 2.0

        # Плотность населения: умеренная — хорошо, пусто — плохо
        if population_density[i] < 0.2:
            d += 1.5
        elif population_density[i] > 0.6:
            d -= 0.5

        # Исторические инциденты — прямой фактор
        d += historical_incidents[i] * 0.05

        # Тип дороги: пешеходные в промзонах опаснее
        if road_type[i] == 3 and district_type[i] == 2:
            d += 1.0

        # Шум
        d += rng.normal(0, 0.5)

        danger_raw[i] = d

    # Масштабирование в диапазон 1-10
    danger_min = danger_raw.min()
    danger_max = danger_raw.max()
    danger_level = 1 + 9 * (danger_raw - danger_min) / (danger_max - danger_min)
    danger_level = np.round(danger_level).astype(int)
    danger_level = np.clip(danger_level, 1, 10)

    # === Сборка DataFrame ===
    df = pd.DataFrame({
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "district_type": district_type,
        "lighting": np.round(lighting, 3),
        "cctv_density": np.round(cctv_density, 3),
        "population_density": np.round(population_density, 3),
        "historical_incidents": historical_incidents,
        "road_type": road_type,
        "danger_level": danger_level,
    })

    # === Балансировка классов ===
    if balance_strategy and balance_strategy != "none":
        print(f"\n[Балансировка] Стратегия: {balance_strategy}")
        print(f"  До балансировки:")
        print(f"    Классов: {df['danger_level'].nunique()}")
        print(f"    Мин примеров: {df['danger_level'].value_counts().min()}")
        print(f"    Макс примеров: {df['danger_level'].value_counts().max()}")

        df = balance_classes(
            df,
            target_col="danger_level",
            strategy=balance_strategy,
            random_state=random_state,
        )

        print(f"  После балансировки:")
        print(f"    Примеров: {len(df)}")
        print(f"    Мин примеров: {df['danger_level'].value_counts().min()}")
        print(f"    Макс примеров: {df['danger_level'].value_counts().max()}")

    # Сохранение в CSV
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n[OK] Данные сохранены: {path}")
        print(f"     Примеров: {len(df)}")
        print(f"     Распределение danger_level:")
        print(df["danger_level"].value_counts().sort_index().to_string())

    return df


def get_feature_names() -> list:
    """Возвращает список названий признаков."""
    return [
        "hour_of_day",
        "day_of_week",
        "district_type",
        "lighting",
        "cctv_density",
        "population_density",
        "historical_incidents",
        "road_type",
    ]


if __name__ == "__main__":
    # Генерация и сохранение датасета с балансировкой
    data_dir = Path(__file__).parent / "data"

    print("=" * 60)
    print("  SafeRoute AI - Генерация синтетических данных")
    print("=" * 60)

    df = generate_synthetic_data(
        n_samples=5000,
        save_path=data_dir / "training_data.csv",
        balance_strategy="oversample",  # oversample / undersample / hybrid / none
    )

    print(f"\nОписания признаков:")
    for name, desc in FEATURE_DESCRIPTIONS.items():
        print(f"  {name}: {desc}")

    print(f"\nСтатистика danger_level:")
    print(f"  Среднее: {df['danger_level'].mean():.2f}")
    print(f"  Медиана: {df['danger_level'].median():.1f}")
    print(f"  Std: {df['danger_level'].std():.2f}")
