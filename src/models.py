"""
Machine Learning Models Module for IoT Intrusion Detection System
Implements the 4 core classification models from Alrefaei & Ilyas (2024):
1. OneVsRest Logistic Regression (OVR-LR)
2. Random Forest Classifier (RF)
3. OneVsRest Decision Tree (OVR-DT)
4. Gradient-Boosted Trees / XGBoost Equivalent (OVR-GBT)

Includes execution timing (training & inference latency) and optional 5-Fold Cross Validation.
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional

try:
    from pyspark.sql import DataFrame
    from pyspark.ml.classification import (
        LogisticRegression,
        RandomForestClassifier,
        DecisionTreeClassifier,
        GBTClassifier,
        OneVsRest,
        OneVsRestModel
    )
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
except ImportError:
    DataFrame = None
    LogisticRegression = None
    RandomForestClassifier = None
    DecisionTreeClassifier = None
    GBTClassifier = None
    OneVsRest = None
    OneVsRestModel = None
    CrossValidator = None
    ParamGridBuilder = None
    MulticlassClassificationEvaluator = None

from src.config import (
    INDEXED_TARGET_COLUMN,
    SELECTED_FEATURES_COL,
    MODEL_CONFIGS,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_model(
    model_key: str,
    features_col: str = SELECTED_FEATURES_COL,
    label_col: str = INDEXED_TARGET_COLUMN,
    custom_params: Optional[Dict[str, Any]] = None
):
    """
    Instantiates a PySpark MLlib classifier configured according to paper specifications.

    Args:
        model_key: Identifier key ('lr', 'rf', 'dt', 'gbt').
        features_col: Name of the input vector column.
        label_col: Name of the target label index column.
        custom_params: Optional parameter overrides.

    Returns:
        Configured PySpark MLlib Estimator.
    """
    params = custom_params or {}

    if model_key == "lr":
        # One-vs-Rest Logistic Regression (Table 2 & 3 in paper)
        base_lr = LogisticRegression(
            featuresCol=features_col,
            labelCol="label",
            maxIter=params.get("maxIter", MODEL_CONFIGS["lr"]["maxIter"]),
            regParam=params.get("regParam", MODEL_CONFIGS["lr"]["regParam"]),
            standardization=True
        )
        return OneVsRest(
            classifier=base_lr,
            featuresCol=features_col,
            labelCol=label_col,
            predictionCol="prediction"
        )

    elif model_key == "rf":
        # Random Forest Classifier (Table 4 & 5 in paper)
        return RandomForestClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            predictionCol="prediction",
            numTrees=params.get("numTrees", MODEL_CONFIGS["rf"]["numTrees"]),
            maxDepth=params.get("maxDepth", MODEL_CONFIGS["rf"]["maxDepth"]),
            seed=params.get("seed", RANDOM_SEED)
        )

    elif model_key == "dt":
        # One-vs-Rest Decision Tree Classifier (Table 6 & 7 in paper)
        base_dt = DecisionTreeClassifier(
            featuresCol=features_col,
            labelCol="label",
            maxDepth=params.get("maxDepth", MODEL_CONFIGS["dt"]["maxDepth"]),
            seed=params.get("seed", RANDOM_SEED)
        )
        return OneVsRest(
            classifier=base_dt,
            featuresCol=features_col,
            labelCol=label_col,
            predictionCol="prediction"
        )

    elif model_key == "gbt":
        # One-vs-Rest Gradient-Boosted Trees (XGBoost distributed equivalent, Table 10 & 11)
        base_gbt = GBTClassifier(
            featuresCol=features_col,
            labelCol="label",
            maxIter=params.get("maxIter", MODEL_CONFIGS["gbt"]["maxIter"]),
            maxDepth=params.get("maxDepth", MODEL_CONFIGS["gbt"]["maxDepth"]),
            seed=params.get("seed", RANDOM_SEED)
        )
        return OneVsRest(
            classifier=base_gbt,
            featuresCol=features_col,
            labelCol=label_col,
            predictionCol="prediction"
        )

    else:
        raise ValueError(f"Unknown model_key '{model_key}'. Supported keys: 'lr', 'rf', 'dt', 'gbt'.")


def train_model(
    estimator,
    train_df: DataFrame
) -> Tuple[Any, float]:
    """
    Fits an estimator on the training DataFrame and measures wall-clock training time.

    Args:
        estimator: PySpark Estimator.
        train_df: Training DataFrame.

    Returns:
        (fitted_model, training_time_seconds)
    """
    logger.info(f"Training model: {estimator.__class__.__name__}...")
    start_time = time.perf_counter()
    model = estimator.fit(train_df)
    training_time = time.perf_counter() - start_time
    logger.info(f"Training finished in {training_time:.4f} seconds.")
    return model, training_time


def predict_model(
    model,
    test_df: DataFrame
) -> Tuple[DataFrame, float]:
    """
    Generates predictions on test_df and measures wall-clock prediction latency.
    Forces action execution via `.count()` to ensure distributed computation is timed.

    Args:
        model: Fitted PySpark ML Transformer / Model.
        test_df: Testing DataFrame.

    Returns:
        (predictions_df, prediction_time_seconds)
    """
    logger.info(f"Generating predictions on test set ({test_df.count():,} rows)...")
    start_time = time.perf_counter()
    predictions = model.transform(test_df).cache()
    
    # Materialize execution to record true distributed latency
    _ = predictions.count()
    prediction_time = time.perf_counter() - start_time
    
    logger.info(f"Prediction finished in {prediction_time:.4f} seconds.")
    return predictions, prediction_time


def run_cross_validation(
    estimator,
    train_df: DataFrame,
    num_folds: int = 5,
    label_col: str = INDEXED_TARGET_COLUMN,
    seed: int = RANDOM_SEED
) -> Tuple[Any, float, float]:
    """
    Performs 5-Fold Cross Validation on the training DataFrame as described in the paper.

    Args:
        estimator: PySpark ML Estimator.
        train_df: Training DataFrame.
        num_folds: Number of CV folds (default: 5).
        label_col: Target label column name.
        seed: Random seed.

    Returns:
        (best_model, cv_mean_accuracy, cv_execution_time_seconds)
    """
    logger.info(f"Starting {num_folds}-Fold Cross Validation...")
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy"
    )

    param_grid = ParamGridBuilder().build()  # Evaluates default architecture

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=seed,
        parallelism=2
    )

    start_time = time.perf_counter()
    cv_model = cv.fit(train_df)
    cv_time = time.perf_counter() - start_time

    # Best accuracy score across folds
    cv_accuracy = float(cv_model.avgMetrics[0]) if cv_model.avgMetrics else 0.0
    logger.info(f"{num_folds}-Fold CV finished in {cv_time:.2f}s with mean accuracy: {cv_accuracy:.4f} ({cv_accuracy:.2%}).")
    return cv_model.bestModel, cv_accuracy, cv_time
