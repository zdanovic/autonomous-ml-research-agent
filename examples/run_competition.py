#!/usr/bin/env python
"""
Comet-Swarm CLI Runner - Управляемый запуск на реальных соревнованиях.

Использование:
    uv run python examples/run_competition.py --platform wundernn --mode debug
    uv run python examples/run_competition.py --platform kaggle --mode full
"""

import argparse
import asyncio
from pathlib import Path

import polars as pl

from comet_swarm.agents import StrategistAgent
from comet_swarm.integrations import ExperimentTracker, get_tracer
from comet_swarm.platforms import KaggleAdapter, SolafuneAdapter, WundernnAdapter
from comet_swarm.tools import (
    correlation_analysis,
    cross_validate,
    lag_features,
    profile_dataset,
    rolling_stats,
)

# =============================================================================
# Configuration
# =============================================================================

PLATFORMS = {
    "wundernn": {
        "adapter": WundernnAdapter,
        "data_format": "parquet",
        "target_cols": ["t0", "t1"],
        "description": "LOB Predictorium (Pearson correlation)",
    },
    "solafune": {
        "adapter": SolafuneAdapter,
        "data_format": "csv",
        "target_cols": ["construction_cost_per_m2_usd"],
        "description": "Construction Cost (RMSLE)",
    },
    "kaggle": {
        "adapter": KaggleAdapter,
        "data_format": "csv",
        "target_cols": ["water_level"],
        "description": "Urban Flood Modelling (R²)",
    },
}

MODES = {
    "debug": {
        "max_iterations": 1,
        "n_folds": 2,
        "sample_size": 1000,
        "description": "Быстрая проверка (1 итерация, 2 фолда, 1K сэмплов)",
    },
    "test": {
        "max_iterations": 3,
        "n_folds": 3,
        "sample_size": 10000,
        "description": "Тестовый запуск (3 итерации, 3 фолда, 10K сэмплов)",
    },
    "full": {
        "max_iterations": 10,
        "n_folds": 5,
        "sample_size": None,  # Use all data
        "description": "Полный запуск (10 итераций, 5-fold CV)",
    },
}


def print_header(text: str):
    """Печать заголовка."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, text: str):
    """Печать шага."""
    print(f"\n[{step}] {text}")


async def run_competition(
    platform: str,
    mode: str,
    data_path: str | None = None,
    skip_download: bool = False,
    dry_run: bool = False,
):
    """
    Запуск агента на соревновании.
    
    Args:
        platform: Платформа (wundernn, solafune, kaggle)
        mode: Режим (debug, test, full)
        data_path: Путь к данным (если не скачивать)
        skip_download: Пропустить загрузку данных
        dry_run: Только показать план без выполнения
    """
    config = PLATFORMS[platform]
    mode_config = MODES[mode]
    
    print_header(f"🚀 Comet-Swarm: {config['description']}")
    print(f"\n📊 Платформа: {platform}")
    print(f"🔧 Режим: {mode} - {mode_config['description']}")
    
    if dry_run:
        print("\n⚠️  DRY RUN - только показываем план, не выполняем")
        print("\nПлан выполнения:")
        print("  1. Инициализация трейсера и эксперимента")
        print("  2. Загрузка данных (если не пропущено)")
        print("  3. EDA: профилирование, корреляции")
        print("  4. Стратегия: генерация гипотез LLM")
        print("  5. Фичи: lag, rolling stats")
        print("  6. Обучение: LightGBM CV")
        print("  7. Оценка: анализ результатов")
        print("  8. (опционально) Сабмит")
        return
    
    # 1. Инициализация
    print_step(1, "Инициализация...")
    tracer = get_tracer()
    tracer.start_trace(f"{platform}_run")
    
    tracker = ExperimentTracker(
        project_name="comet-swarm",
        competition_name=f"{platform}_competition",
    )
    
    adapter = config["adapter"]()
    context = adapter.get_competition_context()
    
    print(f"  ✓ Метрика: {context.metric.value} ({context.metric_direction})")
    print(f"  ✓ Формат сабмита: {context.submission_format}")
    
    # 2. Загрузка данных
    print_step(2, "Загрузка данных...")
    
    data_dir = Path(f"./data/{platform}")
    
    if data_path:
        train_path = Path(data_path)
        print(f"  Используем: {train_path}")
    elif skip_download:
        # Ищем существующие данные
        train_path = data_dir / f"train.{config['data_format']}"
        if not train_path.exists():
            print(f"  ❌ Файл не найден: {train_path}")
            print("  Укажите --data-path или уберите --skip-download")
            return
        print(f"  Найден файл: {train_path}")
    else:
        # Скачиваем данные
        print("  Скачивание данных с платформы...")
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            paths = await adapter.download_data(data_dir)
            train_path = paths.get("train") or paths.get("train_dataset")
            if not train_path:
                print("  ❌ Не удалось скачать данные")
                return
            print(f"  ✓ Скачано: {train_path}")
        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}")
            print("  Попробуйте скачать вручную и указать --data-path")
            return
    
    # 3. Загрузка и сэмплирование
    print_step(3, "Чтение данных...")
    
    if config["data_format"] == "parquet":
        df = pl.read_parquet(train_path)
    else:
        df = pl.read_csv(train_path)
    
    original_size = len(df)
    
    if mode_config["sample_size"] and len(df) > mode_config["sample_size"]:
        df = df.sample(n=mode_config["sample_size"], seed=42)
        print(f"  Сэмплирование: {original_size:,} → {len(df):,} строк")
    else:
        print(f"  Размер: {len(df):,} строк")
    
    # 4. EDA
    print_step(4, "Exploratory Data Analysis...")
    
    target_col = config["target_cols"][0]
    
    # Check if target column exists, fallback to 'target' if not
    if target_col not in df.columns:
        if "target" in df.columns:
            target_col = "target"
            print(f"  ⚠️ Using fallback target column: {target_col}")
        else:
            print(f"  ❌ Target column '{target_col}' not found in data")
            print(f"     Available columns: {df.columns[:10]}...")
            return
    
    correlations = {}  # Initialize before try block
    
    try:
        dataset_info = profile_dataset(df, name=platform, target_column=target_col)
        print(f"  ✓ Колонок: {dataset_info.num_columns}")
        print(f"  ✓ Память: {dataset_info.memory_mb:.2f} MB")
        
        correlations = correlation_analysis(df, target_column=target_col, top_n=5)
        print(f"  ✓ Топ корреляции с {target_col}:")
        for feat, corr in list(correlations.items())[:3]:
            print(f"      {feat}: {corr:.4f}")
    except Exception as e:
        print(f"  ⚠️ EDA ошибка: {e}")
        dataset_info = None
    
    # 5. Стратегия
    print_step(5, "Генерация стратегии (LLM)...")
    
    strategist = StrategistAgent()
    
    dataset_summary = f"{len(df)} строк, {len(df.columns)} колонок"
    eda_summary = "Топ корреляции: " + ", ".join(
        [f"{k}={v:.3f}" for k, v in list(correlations.items())[:3]]
    ) if correlations else "Нет данных"
    
    try:
        strategy = await strategist.generate_strategy(
            competition_context=context.model_dump(),
            dataset_info=dataset_summary,
            eda_insights=eda_summary,
        )
        print(f"  ✓ Приоритет: {strategy.priority[:80]}...")
        print(f"  ✓ Гипотезы: {len(strategy.hypotheses)}")
    except Exception as e:
        print(f"  ❌ Ошибка стратегии: {e}")
        return
    
    # 6. Feature Engineering
    print_step(6, "Feature Engineering...")
    
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric() and c not in config["target_cols"]]
    
    if numeric_cols:
        try:
            # Найти группировочную колонку (если есть)
            group_col = None
            for candidate in ["seq_ix", "id", "building_id", "node_id"]:
                if candidate in df.columns:
                    group_col = candidate
                    break
            
            # Lag features
            feature_col = numeric_cols[0]
            if group_col:
                df = lag_features(df, column=feature_col, lags=[1, 3], group_by=group_col)
                df = rolling_stats(df, column=feature_col, windows=[3], stats=["mean"], group_by=group_col)
            else:
                df = lag_features(df, column=feature_col, lags=[1, 3])
                df = rolling_stats(df, column=feature_col, windows=[3], stats=["mean"])
            
            new_features = [c for c in df.columns if "lag" in c or "roll" in c]
            print(f"  ✓ Новых фич: {len(new_features)}")
        except Exception as e:
            print(f"  ⚠️ Feature engineering ошибка: {e}")
    else:
        print("  ⚠️ Нет числовых колонок для фичей")
    
    # 7. Training
    print_step(7, f"Model Training ({mode_config['n_folds']}-fold CV)...")
    
    # Exclude target_col from features (avoid duplicate in select)
    feature_cols = [c for c in df.columns if c not in config["target_cols"] and c != target_col and df[c].dtype.is_numeric()]
    
    if not feature_cols or target_col not in df.columns:
        print("  ❌ Нет фичей или таргета для обучения")
        return
    
    # Очистка от NaN
    df_train = df.select(feature_cols + [target_col]).drop_nulls()
    
    if len(df_train) < 100:
        print(f"  ❌ Слишком мало данных после очистки: {len(df_train)}")
        return
    
    X = df_train.select(feature_cols).to_numpy()
    y = df_train.select(target_col).to_numpy().flatten()
    
    print(f"  Сэмплов: {len(X):,}")
    print(f"  Фичей: {len(feature_cols)}")
    
    # Запуск эксперимента
    tracker.start_experiment(
        name=f"{platform}_{mode}_v1",
        tags=[platform, mode, "baseline"],
    )
    
    try:
        cv_result = cross_validate(
            X, y,
            model_fn="lightgbm",
            params={"num_leaves": 31, "learning_rate": 0.05, "verbose": -1},
            n_folds=mode_config["n_folds"],
            log_to_comet=True,
        )
        
        print(f"\n  ✓ CV Mean: {cv_result['cv_mean']:.4f}")
        print(f"  ✓ CV Std: {cv_result['cv_std']:.4f}")
        print(f"  ✓ Fold scores: {[f'{s:.4f}' for s in cv_result['fold_scores']]}")
        
        tracker.log_metrics({
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
        })
        
    except Exception as e:
        print(f"  ❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Завершение эксперимента
    experiment_result = tracker.end_experiment()
    if experiment_result:
        print(f"\n  📊 Experiment: {experiment_result.experiment_url}")
    
    # 8. Бюджет
    print_step(8, "Budget Summary")
    budget = tracer.get_budget_status()
    print(f"  💰 Потрачено: ${budget.total_spent_usd:.4f}")
    print(f"  💰 Осталось: ${budget.remaining_usd:.2f}")
    print(f"  💰 Использовано: {budget.percentage_used:.1f}%")
    
    print_header("✅ Запуск завершён!")
    
    # Подсказки
    print("\n💡 Следующие шаги:")
    if mode == "debug":
        print("  → Убедитесь, что всё работает")
        print("  → Запустите с --mode test для более полного теста")
    elif mode == "test":
        print("  → Проверьте результаты в Comet ML")
        print("  → Запустите с --mode full для боевого прогона")
    else:
        print("  → Проанализируйте результаты в Comet ML")
        print("  → Сделайте сабмит, если результаты хорошие")


def main():
    parser = argparse.ArgumentParser(
        description="Comet-Swarm: Запуск на реальных соревнованиях",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Быстрая проверка на Wundernn
  uv run python examples/run_competition.py --platform wundernn --mode debug

  # Тестовый запуск на Kaggle
  uv run python examples/run_competition.py --platform kaggle --mode test

  # Полный запуск с своими данными
  uv run python examples/run_competition.py --platform solafune --mode full --data-path ./data/train.csv

  # Dry run (только показать план)
  uv run python examples/run_competition.py --platform wundernn --dry-run
        """,
    )
    
    parser.add_argument(
        "--platform", "-p",
        choices=list(PLATFORMS.keys()),
        required=True,
        help="Платформа соревнования",
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=list(MODES.keys()),
        default="debug",
        help="Режим запуска (default: debug)",
    )
    
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Путь к файлу данных (вместо скачивания)",
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Использовать ранее скачанные данные",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать план без выполнения",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_competition(
        platform=args.platform,
        mode=args.mode,
        data_path=args.data_path,
        skip_download=args.skip_download,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
