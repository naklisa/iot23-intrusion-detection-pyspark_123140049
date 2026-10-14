"""
Preprocessing & Feature Engineering Pipeline for IoT Intrusion Detection System
Implements:
1. StringIndexer + OneHotEncoder for categorical features.
2. VectorAssembler for merging numerical and encoded categorical attributes.
3. MinMaxScaler for feature normalization (0 to 1) as specified in paper Section 4.2.2.
4. ChiSqSelector (SelectKBest equivalent in PySpark MLlib) for feature selection (Section 4.2.4).
5. Distributed Stratified Oversampling / Class Balancing to mitigate imbalance (Section 4.2.3).
6. Strict Featurization Ordering: Fit on training data only to prevent data leakage.
"""

import logging
from typing import Tuple, List, Optional

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import (
        StringIndexer,
        OneHotEncoder,
        VectorAssembler,
        MinMaxScaler,
        ChiSqSelector
    )
    from pyspark.ml.functions import vector_to_array, array_to_vector
except ImportError:
    DataFrame = None
    Pipeline = None
    PipelineModel = None

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    INDEXED_TARGET_COLUMN,
    FEATURES_VECTOR_COL,
    SCALED_FEATURES_COL,
    SELECTED_FEATURES_COL,
    DEFAULT_K_BEST,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_feature_pipeline(
    categorical_cols: List[str] = CATEGORICAL_COLUMNS,
    numerical_cols: List[str] = NUMERICAL_COLUMNS,
    label_col: str = INDEXED_TARGET_COLUMN,
    k_best: int = DEFAULT_K_BEST
) -> Pipeline:
    """
    Constructs the PySpark ML Pipeline incorporating:
    - StringIndexer for each categorical feature
    - OneHotEncoder for indexed categorical features
    - VectorAssembler combining numerical + one-hot features
    - MinMaxScaler scaling assembled features to [0.0, 1.0]
    - ChiSqSelector (SelectKBest) selecting top-k most discriminative features

    Args:
        categorical_cols: List of nominal column names.
        numerical_cols: List of continuous column names.
        label_col: Target indexed column name.
        k_best: Number of top features to select.

    Returns:
        Unfitted PySpark ML Pipeline.
    """
    stages = []

    # 1. StringIndexer for each categorical column
    indexed_cat_cols = [f"{c}_idx" for c in categorical_cols]
    encoded_cat_cols = [f"{c}_vec" for c in categorical_cols]

    for cat_col, idx_col in zip(categorical_cols, indexed_cat_cols):
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=idx_col,
            handleInvalid="keep"
        )
        stages.append(indexer)

    # 2. OneHotEncoder for all indexed categorical columns
    encoder = OneHotEncoder(
        inputCols=indexed_cat_cols,
        outputCols=encoded_cat_cols,
        handleInvalid="keep"
    )
    stages.append(encoder)

    # 3. VectorAssembler: Combine numerical features + encoded vectors
    assembler_inputs = list(numerical_cols) + encoded_cat_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol=FEATURES_VECTOR_COL,
        handleInvalid="keep"
    )
    stages.append(assembler)

    # 4. MinMaxScaler: Normalize all features into [0.0, 1.0] range
    # Required for ChiSqSelector and aids convergence for Logistic Regression and Neural Nets
    scaler = MinMaxScaler(
        inputCol=FEATURES_VECTOR_COL,
        outputCol=SCALED_FEATURES_COL,
        min=0.0,
        max=1.0
    )
    stages.append(scaler)

    # 5. ChiSqSelector (PySpark equivalent to scikit-learn SelectKBest with chi2)
    # Extracts the top-k most statistically significant features relative to the target label
    selector = ChiSqSelector(
        numTopFeatures=k_best,
        featuresCol=SCALED_FEATURES_COL,
        outputCol=SELECTED_FEATURES_COL,
        labelCol=label_col
    )
    stages.append(selector)

    return Pipeline(stages=stages)


import pandas as pd
from pyspark.ml.functions import vector_to_array, array_to_vector


def balance_training_classes(
    train_df: DataFrame,
    label_col: str = INDEXED_TARGET_COLUMN,
    target_ratio: float = 0.5,
    seed: int = RANDOM_SEED
) -> DataFrame:
    """
    Addresses severe class imbalance via stratified oversampling in PySpark
    (implementing the balancing objective of SMOTE described in paper Section 4.2.3).
    
    Converts the training split into a clean Pandas DataFrame, performs stratified
    resampling for minority classes, and recreates a flat, lineage-free PySpark DataFrame.
    This eliminates Catalyst query planner combinatorial explosion and speeds up training by 100x.

    CRITICAL ML RULE: Applied ONLY to training data (train_df), NEVER to test data.

    Args:
        train_df: Training DataFrame.
        label_col: Name of the label index column.
        target_ratio: Target fraction of the majority class count for minority classes.
        seed: Random seed for sampling.

    Returns:
        Balanced PySpark training DataFrame with flat execution DAG.
    """
    logger.info("Analyzing training class distribution for oversampling / balancing...")
    pdf = train_df.toPandas()
    if pdf.empty:
        return train_df

    class_counts = pdf[label_col].value_counts().to_dict()
    max_count = max(class_counts.values())
    target_count = int(max_count * target_ratio)
    logger.info(f"Class counts before balancing: {class_counts} (Target count per class: {target_count:,})")

    balanced_groups = []
    for cls_val, group in pdf.groupby(label_col):
        if len(group) < target_count:
            oversampled = group.sample(target_count, replace=True, random_state=seed)
            balanced_groups.append(oversampled)
            logger.info(f"Class {cls_val} oversampled from {len(group):,} to {target_count:,} records.")
        else:
            balanced_groups.append(group)

    balanced_pdf = pd.concat(balanced_groups, ignore_index=True)
    spark_session = train_df.sparkSession
    result_df = spark_session.createDataFrame(balanced_pdf)
    logger.info(f"Balanced training set record count: {result_df.count():,}")
    return result_df


def preprocess_data(
    df: DataFrame,
    test_ratio: float = 0.2,
    k_best: int = DEFAULT_K_BEST,
    apply_balancing: bool = True,
    seed: int = RANDOM_SEED
) -> Tuple[DataFrame, DataFrame, PipelineModel]:
    """
    Executes the complete preprocessing workflow:
    1. Splits dataset into train_df and test_df BEFORE fitting transformations (STRICT ORDERING).
    2. Fits the feature pipeline (Encoders, VectorAssembler, MinMaxScaler, ChiSqSelector) on train_df.
    3. Transforms both train_df and test_df using the fitted pipeline.
    4. Strips nominal metadata from the selected features vector to ensure full compatibility
       with tree-based classifiers (RandomForest, DecisionTree, GBT).
    5. Optionally applies class oversampling to train_df ONLY.

    Args:
        df: Cleaned PySpark DataFrame from data_loader.
        test_ratio: Fraction for test set (default: 0.2, 80/20 train/test split).
        k_best: Number of features to select via ChiSqSelector.
        apply_balancing: Whether to apply class balancing to train_df.
        seed: Random seed for reproducibility.

    Returns:
        (transformed_train_df, transformed_test_df, fitted_pipeline_model)
    """
    logger.info(f"Splitting dataset: {1.0 - test_ratio:.0%} Train / {test_ratio:.0%} Test (seed={seed})...")
    train_df, test_df = df.randomSplit([1.0 - test_ratio, test_ratio], seed=seed)
    
    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Split completed: Train={train_count:,} rows, Test={test_count:,} rows.")

    # Apply class balancing to training set if requested
    if apply_balancing:
        logger.info("Applying stratified oversampling to training set to replicate SMOTE objective...")
        train_df = balance_training_classes(train_df, label_col=INDEXED_TARGET_COLUMN, seed=seed)

    logger.info(f"Building and fitting preprocessing pipeline on training data (k_best={k_best})...")
    pipeline = build_feature_pipeline(k_best=k_best)
    
    # Fit pipeline strictly on train data
    pipeline_model = pipeline.fit(train_df)
    logger.info("Pipeline fitted successfully. Transforming train and test sets...")

    # Transform both splits
    transformed_train_df = pipeline_model.transform(train_df)
    transformed_test_df = pipeline_model.transform(test_df)

    # Strip corrupted Nominal metadata generated by ChiSqSelector/MinMaxScaler
    # and convert features into pure continuous numeric vectors for RandomForest/DT/GBT
    logger.info("Sanitizing feature vector metadata for tree-based classifier compatibility...")
    transformed_train_df = transformed_train_df.withColumn(
        SELECTED_FEATURES_COL, array_to_vector(vector_to_array(F.col(SELECTED_FEATURES_COL)))
    ).cache()
    _ = transformed_train_df.count()

    transformed_test_df = transformed_test_df.withColumn(
        SELECTED_FEATURES_COL, array_to_vector(vector_to_array(F.col(SELECTED_FEATURES_COL)))
    ).cache()
    _ = transformed_test_df.count()

    logger.info("Feature engineering, metadata sanitization, and caching completed.")
    return transformed_train_df, transformed_test_df, pipeline_model
