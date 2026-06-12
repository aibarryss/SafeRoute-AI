"""
Обучение модели предсказания уровня опасности района.

Модель: Gradient Boosting Classifier (scikit-learn)
Признаки: 8 городских признаков (см. data_generator.py)
Цель: предсказание danger_level (1-10)

Метрики валидации:
- Accuracy: общая точность классификации
- Precision: точность по каждому классу (macro/weighted)
- Recall: полнота по каждому классу (macro/weighted)
- F1-score: гармоническое среднее precision и recall
- MAE/RMSE: ошибки регрессии (danger_level как число)
- Cross-validation: 5-fold CV для стабильности

Визуализации:
- Feature importance bar chart
- Confusion matrix
- Classification report

Использование:
    python -m backend.ml.train_model

Или как модуль:
    from ml.train_model import train_and_save_model
    metrics = train_and_save_model()
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
import joblib

# Добавляем родительскую директорию для импорта data_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_generator import (
    generate_synthetic_data,
    get_feature_names,
    FEATURE_DESCRIPTIONS,
)


# Пути
ML_DIR = Path(__file__).parent
MODEL_DIR = ML_DIR / "models"
DATA_DIR = ML_DIR / "data"
MODEL_PATH = MODEL_DIR / "danger_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
REPORTS_DIR = ML_DIR / "reports"


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Подготавливает признаки и целевую переменную.

    Returns:
        X (признаки), y (целевая переменная), feature_names
    """
    feature_names = get_feature_names()
    X = df[feature_names].values
    y = df["danger_level"].values
    return X, y, feature_names


def plot_feature_importance(
    model: GradientBoostingClassifier,
    feature_names: list,
    save_path: Path = None,
    show: bool = False,
) -> Optional[str]:
    """
    Создаёт визуализацию важности признаков.

    Args:
        model: обученная модель
        feature_names: названия признаков
        save_path: путь для сохранения изображения
        show: показывать ли график

    Returns:
        Путь к сохранённому изображению или None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Backend без GUI

        # Получение важности признаков
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar chart
        bars = ax.bar(range(len(feature_names)), importances[indices], color='steelblue')

        # Настройка осей
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance - Danger Level Prediction', fontsize=14, fontweight='bold')

        # Подписи
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')

        # Добавление значений на bars
        for bar, importance in zip(bars, importances[indices]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{importance:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)

        # Сохранение
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Feature importance сохранён: {save_path}")

        if show:
            plt.show()

        plt.close()
        return str(save_path) if save_path else None

    except ImportError:
        print("  [WARNING] matplotlib не установлен, визуализация пропущена")
        return None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None,
    show: bool = False,
) -> Optional[str]:
    """
    Создаёт визуализацию матрицы ошибок.

    Args:
        y_true: истинные значения
        y_pred: предсказанные значения
        save_path: путь для сохранения изображения
        show: показывать ли график

    Returns:
        Путь к сохранённому изображению или None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # Вычисление матрицы ошибок
        cm = confusion_matrix(y_true, y_pred)

        # Создание графика
        fig, ax = plt.subplots(figsize=(10, 8))

        # Heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Настройка осей
        classes = sorted(np.unique(y_true))
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True Label',
            xlabel='Predicted Label',
            title='Confusion Matrix - Danger Level Prediction'
        )

        # Поворот подписей
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Добавление чисел в ячейки
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8
                )

        plt.tight_layout()

        # Сохранение
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Confusion matrix сохранена: {save_path}")

        if show:
            plt.show()

        plt.close()
        return str(save_path) if save_path else None

    except ImportError:
        print("  [WARNING] matplotlib не установлен, визуализация пропущена")
        return None


def train_and_save_model(
    n_samples: int = 5000,
    random_state: int = 42,
    verbose: bool = True,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Обучает модель Gradient Boosting и сохраняет её.

    Процесс:
    1. Генерация синтетических данных
    2. Разделение на train/test (80/20)
    3. Обучение GradientBoostingClassifier
    4. Оценка на тестовой выборке
    5. Cross-validation (5-fold)
    6. Сохранение модели, метрик и визуализаций

    Args:
        n_samples: количество обучающих примеров
        random_state: seed для воспроизводимости
        verbose: выводить ли прогресс
        save_plots: сохранять ли визуализации

    Returns:
        Словарь с метриками модели
    """
    if verbose:
        print("=" * 70)
        print("  SafeRoute AI - Обучение модели предсказания опасности")
        print("=" * 70)

    # 1. Генерация данных
    if verbose:
        print("\n[1/6] Генерация синтетических данных...")

    df = generate_synthetic_data(
        n_samples=n_samples,
        random_state=random_state,
        save_path=DATA_DIR / "training_data.csv",
        balance_strategy="oversample",
    )

    # 2. Подготовка признаков
    if verbose:
        print(f"\n[2/6] Подготовка признаков ({len(get_feature_names())} фичей)...")

    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"  Train: {len(X_train)} примеров")
        print(f"  Test:  {len(X_test)} примеров")
        print(f"  Классов: {len(np.unique(y))} (1-10)")

    # 3. Обучение модели
    if verbose:
        print("\n[3/6] Обучение Gradient Boosting Classifier...")
        print("  Параметры:")
        print("    - n_estimators: 200")
        print("    - max_depth: 5")
        print("    - learning_rate: 0.1")
        print("    - subsample: 0.8")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        verbose=0,
    )

    model.fit(X_train, y_train)

    # 4. Оценка модели на тестовой выборке
    if verbose:
        print("\n[4/6] Оценка модели на тестовой выборке...")

    y_pred = model.predict(X_test)

    # Основные метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_macro": round(recall_macro, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": len(feature_names),
        "n_classes": len(np.unique(y)),
    }

    # Hyperparameters
    metrics["hyperparameters"] = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "subsample": model.subsample,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
    }

    if verbose:
        print(f"\n  Результаты:")
        print(f"  Accuracy:          {metrics['accuracy']:.2%}")
        print(f"  Precision (macro): {metrics['precision_macro']:.2%}")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.2%}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.2%}")
        print(f"  Recall (weighted): {metrics['recall_weighted']:.2%}")
        print(f"  F1-score (macro):  {metrics['f1_macro']:.2%}")
        print(f"  F1-score (weighted): {metrics['f1_weighted']:.2%}")
        print(f"  MAE:               {metrics['mae']:.3f}")
        print(f"  RMSE:              {metrics['rmse']:.3f}")

    # 5. Cross-validation
    if verbose:
        print("\n[5/6] Cross-validation (5-fold)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    metrics["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_accuracy_std"] = round(cv_scores.std(), 4)
    metrics["cv_scores"] = [round(s, 4) for s in cv_scores]

    if verbose:
        print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.2%} (+/- {metrics['cv_accuracy_std']:.2%})")
        print(f"  Folds: {[f'{s:.2%}' for s in cv_scores]}")

    # Feature importance
    feature_importance = dict(zip(
        feature_names,
        [round(float(x), 4) for x in model.feature_importances_]
    ))
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    metrics["feature_importance"] = feature_importance

    if verbose:
        print(f"\n  Важность признаков:")
        for name, importance in feature_importance.items():
            bar = "#" * int(importance * 50)
            desc = FEATURE_DESCRIPTIONS.get(name, "")
            print(f"    {name:25s} {importance:.4f}  {bar}")
            if verbose and desc:
                print(f"      -> {desc}")

    # Classification report
    if verbose:
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Сохранение модели, метрик и визуализаций
    if verbose:
        print("\n[6/6] Сохранение результатов...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Сохранение модели
    joblib.dump(model, MODEL_PATH)
    if verbose:
        print(f"  Модель сохранена: {MODEL_PATH}")

    # Сохранение метрик
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  Метрики сохранены: {METRICS_PATH}")

    # Визуализации
    if save_plots:
        if verbose:
            print("\n  Генерация визуализаций...")

        # Feature importance
        plot_feature_importance(
            model,
            feature_names,
            save_path=REPORTS_DIR / "feature_importance.png",
            show=False,
        )

        # Confusion matrix
        plot_confusion_matrix(
            y_test,
            y_pred,
            save_path=REPORTS_DIR / "confusion_matrix.png",
            show=False,
        )

    return metrics


def load_model():
    """Загружает обученную модель с диска."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}\n"
            f"Запустите обучение: python -m backend.ml.train_model"
        )
    return joblib.load(MODEL_PATH)


def evaluate_model_quality(metrics: Dict[str, Any]) -> str:
    """
    Оценивает качество модели на основе метрик.

    Returns:
        Строка с оценкой: "excellent", "good", "acceptable", "poor"
    """
    accuracy = metrics.get("accuracy", 0)
    f1_macro = metrics.get("f1_macro", 0)
    cv_mean = metrics.get("cv_accuracy_mean", 0)

    # Среднее значение ключевых метрик
    avg_score = (accuracy + f1_macro + cv_mean) / 3

    if avg_score >= 0.85:
        return "excellent"
    elif avg_score >= 0.75:
        return "good"
    elif avg_score >= 0.65:
        return "acceptable"
    else:
        return "poor"


if __name__ == "__main__":
    metrics = train_and_save_model(
        n_samples=5000,
        verbose=True,
        save_plots=True,
    )

    # Оценка качества
    quality = evaluate_model_quality(metrics)

    print("\n" + "=" * 70)
    print("  Обучение завершено!")
    print(f"  Качество модели: {quality.upper()}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1-macro: {metrics['f1_macro']:.2%}")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.2%} (+/- {metrics['cv_accuracy_std']:.2%})")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metrics: {METRICS_PATH}")
    print(f"  Reports: {REPORTS_DIR}")
    print("=" * 70)
