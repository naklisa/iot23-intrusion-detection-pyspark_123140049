"""
Data Loader & Pre-cleaning Module for IoT Intrusion Detection System
Provides:
1. get_spark_session: Initializes and optimizes local Apache Spark session.
2. load_and_clean_data: Ingests, sanitizes, and normalizes the IoT-23 CSV dataset.
3. generate_synthetic_data: Generates synthetic IoT-23 data for smoke testing.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType, LongType
    )
except ImportError:
    # Allow importing module even before PySpark is loaded
    SparkSession = None
    DataFrame = None
    F = None

from src.config import (
    DEFAULT_DATA_FILE,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    LABEL_STANDARDIZATION_MAP,
    COLUMN_RENAME_MAP,
    DROP_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    INDEXED_TARGET_COLUMN,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def configure_java_environment():
    """
    Auto-detects OpenJDK / JRE installation on Windows/Linux if JAVA_HOME is not set,
    and prepends the Java bin directory to PATH for seamless Spark initialization.
    Also binds PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to sys.executable.
    """
    import os
    # Ensure Spark Python workers use the current active python interpreter (critical for Windows)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if "JAVA_HOME" not in os.environ or not Path(os.environ["JAVA_HOME"]).exists():
        candidates = [
            Path("C:/Program Files/Microsoft/jdk-17.0.20.101-hotspot"),
            Path("C:/Program Files/Microsoft"),
            Path("C:/Program Files/Java"),
            Path("C:/Program Files/Eclipse Adoptium")
        ]
        found_java_home = None
        for cand in candidates:
            if cand.exists():
                if (cand / "bin" / "java.exe").exists() or (cand / "bin" / "java").exists():
                    found_java_home = cand
                    break
                for sub in cand.glob("*jdk*"):
                    if (sub / "bin" / "java.exe").exists() or (sub / "bin" / "java").exists():
                        found_java_home = sub
                        break
                if found_java_home:
                    break
        if found_java_home:
            os.environ["JAVA_HOME"] = str(found_java_home)
            java_bin = str(found_java_home / "bin")
            if java_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{java_bin};{os.environ.get('PATH', '')}"
            logger.info(f"Auto-configured JAVA_HOME -> {found_java_home}")


def get_spark_session(
    app_name: str = "IoT23_IDS_Classification",
    memory: str = "4g"
) -> SparkSession:
    """
    Initializes and returns an optimized local SparkSession for PySpark MLlib.
    Configures in-memory execution, Arrow serialization, and minimal driver overhead.
    """
    configure_java_environment()
    if SparkSession is None:
        raise ImportError(
            "pyspark is not installed in the active environment. "
            "Please ensure requirements.txt is installed."
        )

    logger.info(f"Initializing Apache Spark Session (app: '{app_name}', memory: {memory})...")
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
    )
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    logger.info(f"Apache Spark version {spark.version} successfully initialized.")
    return spark


def load_and_clean_data(
    spark: SparkSession,
    file_path: Optional[Union[str, Path]] = None,
    sample_ratio: Optional[float] = None,
    seed: int = RANDOM_SEED
) -> DataFrame:
    """
    Loads the IoT-23 combined dataset CSV into a PySpark DataFrame, cleans
    missing values, renames columns to valid Spark identifiers, standardizes
    labels into the 5 target classes, and casts numeric columns.

    Args:
        spark: Active SparkSession.
        file_path: Path to `iot23_combined.csv`. Defaults to `data/iot23_combined.csv`.
        sample_ratio: Optional fraction (0.0 < ratio <= 1.0) to subsample for faster execution.
        seed: Random seed for sampling.

    Returns:
        Cleaned PySpark DataFrame ready for feature engineering.
    """
    if file_path is None:
        file_path = DEFAULT_DATA_FILE
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'.\n"
            f"Please download 'iot23_combined.csv' from Kaggle as instructed in 'data/README_DATA.md',\n"
            f"or run the pipeline in smoke test mode: `python main.py --mode smoke`."
        )

    logger.info(f"Reading dataset from: {file_path}")
    # Read CSV with header and inferred schema
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")  # Read as string first to safely clean '-' placeholders
        .csv(str(file_path))
    )

    initial_count = df.count()
    logger.info(f"Raw record count: {initial_count:,} rows across {len(df.columns)} columns.")

    # 1. Rename dot-notated columns to valid underscore identifiers
    for old_col, new_col in COLUMN_RENAME_MAP.items():
        if old_col in df.columns:
            df = df.withColumnRenamed(old_col, new_col)

    # 2. Drop non-predictive columns and identifiers
    existing_drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if existing_drop_cols:
        logger.info(f"Dropping non-predictive columns: {existing_drop_cols}")
        df = df.drop(*existing_drop_cols)

    # 3. Clean and cast numerical columns: replace '-' or empty with 0.0
    for num_col in NUMERICAL_COLUMNS:
        if num_col in df.columns:
            df = df.withColumn(
                num_col,
                F.when(
                    (F.col(num_col) == "-") | (F.col(num_col).isNull()) | (F.trim(F.col(num_col)) == ""),
                    0.0
                ).otherwise(F.col(num_col).cast(DoubleType()))
            )
            # Impute any resulting NaN / Null with 0.0
            df = df.fillna({num_col: 0.0})

    # 4. Clean categorical columns: replace '-' with 'Unknown'
    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in df.columns:
            df = df.withColumn(
                cat_col,
                F.when(
                    (F.col(cat_col) == "-") | (F.col(cat_col).isNull()) | (F.trim(F.col(cat_col)) == ""),
                    "Unknown"
                ).otherwise(F.trim(F.col(cat_col)))
            )

    # 5. Standardize target label into 5 multiclass categories
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {df.columns}")

    # Build mapping expression using SQL CASE WHEN
    norm_label_col = F.lower(F.trim(F.col(TARGET_COLUMN)))
    
    # Conditional ladder for the 5 classes
    mapping_expr = (
        F.when(norm_label_col.contains("benign"), "Benign")
        .when(norm_label_col.contains("ddos"), "DDoS")
        .when(norm_label_col.contains("okiru"), "Okiru")
        .when(
            norm_label_col.contains("horizontal") | 
            norm_label_col.contains("portscan") | 
            norm_label_col.contains("partofahorizontalportscan"),
            "PartOfAHorizontalPortScan"
        )
        .when(
            norm_label_col.contains("attack") | 
            norm_label_col.contains("c&c") | 
            norm_label_col.contains("filedownload"),
            "Attack"
        )
        .otherwise("Unknown")
    )

    df = df.withColumn("clean_label", mapping_expr)

    # Filter out records that could not be mapped to the 5 designated classes
    df = df.filter(F.col("clean_label") != "Unknown")

    # Add numeric target index (0 to 4)
    index_expr = (
        F.when(F.col("clean_label") == "Benign", 0)
        .when(F.col("clean_label") == "DDoS", 1)
        .when(F.col("clean_label") == "Okiru", 2)
        .when(F.col("clean_label") == "PartOfAHorizontalPortScan", 3)
        .when(F.col("clean_label") == "Attack", 4)
        .otherwise(0)
    ).cast(IntegerType())

    df = df.withColumn(INDEXED_TARGET_COLUMN, index_expr)

    # 6. Optional subsampling for development or quick demonstration
    if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
        logger.info(f"Subsampling dataset at ratio: {sample_ratio:.2%} (seed={seed})...")
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=seed)

    final_count = df.count()
    logger.info(f"Cleaned dataset ready: {final_count:,} records.")
    return df


def generate_synthetic_data(
    spark: SparkSession,
    n_samples: int = 5000,
    seed: int = RANDOM_SEED
) -> DataFrame:
    """
    Generates a realistic synthetic IoT-23 dataset in PySpark.
    Mimics feature distributions and 5 multiclass attack labels for smoke testing
    and pipeline validation without requiring the full 1.44M row Kaggle CSV.

    Args:
        spark: Active SparkSession.
        n_samples: Number of synthetic rows to generate (default: 5000).
        seed: Random seed for reproducibility.

    Returns:
        PySpark DataFrame matching the schema of `load_and_clean_data`.
    """
    logger.info(f"Generating synthetic IoT-23 dataset ({n_samples:,} records, seed={seed})...")
    np.random.seed(seed)

    # Proportions roughly matching IoT-23 combined distribution:
    # PartOfAHorizontalPortScan: ~55%, Okiru: ~20%, Benign: ~15%, DDoS: ~9%, Attack: ~1%
    classes = [0, 1, 2, 3, 4]
    class_probs = [0.15, 0.10, 0.20, 0.50, 0.05]
    
    label_indices = np.random.choice(classes, size=n_samples, p=class_probs)
    clean_labels = [CLASS_NAMES[idx] for idx in label_indices]

    # Generate synthetic features conditioned on attack classes
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "dns", "dhcp", "ssl", "Unknown"]
    conn_states = ["S0", "SF", "REJ", "RSTO", "RSTR", "S1", "Unknown"]
    histories = ["ShADF", "D", "Sr", "ShA", "S", "Unknown"]

    data = []
    for i in range(n_samples):
        cls_idx = label_indices[i]
        
        # Characteristic adjustments per attack type
        if cls_idx == 0:  # Benign
            proto = np.random.choice(protocols, p=[0.7, 0.25, 0.05])
            service = np.random.choice(services, p=[0.4, 0.3, 0.1, 0.1, 0.1])
            conn_state = np.random.choice(conn_states, p=[0.05, 0.8, 0.05, 0.05, 0.02, 0.02, 0.01])
            duration = max(0.001, float(np.random.exponential(scale=1.5)))
            orig_bytes = float(np.random.lognormal(mean=6.0, sigma=1.2))
            resp_bytes = float(np.random.lognormal(mean=7.0, sigma=1.5))
            orig_pkts = int(max(1, np.random.poisson(lam=12)))
            resp_pkts = int(max(1, np.random.poisson(lam=14)))
            id_orig_p = int(np.random.randint(1024, 65535))
            id_resp_p = int(np.random.choice([80, 443, 53, 123, 8080]))
        elif cls_idx == 1:  # DDoS
            proto = "tcp"
            service = "Unknown"
            conn_state = "S0"
            duration = max(0.0001, float(np.random.exponential(scale=0.05)))
            orig_bytes = float(np.random.uniform(0, 100))
            resp_bytes = 0.0
            orig_pkts = int(max(1, np.random.poisson(lam=3)))
            resp_pkts = 0
            id_orig_p = int(np.random.randint(1024, 65535))
            id_resp_p = int(np.random.choice([80, 23, 8080]))
        elif cls_idx == 2:  # Okiru
            proto = np.random.choice(["tcp", "udp"], p=[0.8, 0.2])
            service = "Unknown"
            conn_state = np.random.choice(["S0", "REJ"], p=[0.7, 0.3])
            duration = max(0.0001, float(np.random.exponential(scale=0.1)))
            orig_bytes = float(np.random.uniform(0, 50))
            resp_bytes = 0.0
            orig_pkts = int(max(1, np.random.poisson(lam=2)))
            resp_pkts = 0
            id_orig_p = int(np.random.randint(1024, 65535))
            id_resp_p = int(np.random.choice([23, 2323, 7547, 5555]))
        elif cls_idx == 3:  # PartOfAHorizontalPortScan
            proto = "tcp"
            service = "Unknown"
            conn_state = "S0"
            duration = 0.0
            orig_bytes = 0.0
            resp_bytes = 0.0
            orig_pkts = 1
            resp_pkts = 0
            id_orig_p = int(np.random.randint(1024, 65535))
            id_resp_p = int(np.random.randint(1, 1024))
        else:  # Attack / General
            proto = np.random.choice(protocols, p=[0.8, 0.15, 0.05])
            service = np.random.choice(services, p=[0.3, 0.2, 0.1, 0.1, 0.3])
            conn_state = np.random.choice(conn_states, p=[0.3, 0.4, 0.1, 0.1, 0.05, 0.03, 0.02])
            duration = max(0.001, float(np.random.exponential(scale=2.0)))
            orig_bytes = float(np.random.lognormal(mean=5.0, sigma=1.0))
            resp_bytes = float(np.random.lognormal(mean=5.5, sigma=1.2))
            orig_pkts = int(max(1, np.random.poisson(lam=8)))
            resp_pkts = int(max(0, np.random.poisson(lam=6)))
            id_orig_p = int(np.random.randint(1024, 65535))
            id_resp_p = int(np.random.choice([21, 22, 23, 80, 443, 3389]))

        history = np.random.choice(histories)
        missed_bytes = 0.0
        orig_ip_bytes = float(orig_pkts * 40 + orig_bytes)
        resp_ip_bytes = float(resp_pkts * 40 + resp_bytes)

        data.append({
            "proto": str(proto),
            "service": str(service),
            "conn_state": str(conn_state),
            "history": str(history),
            "id_orig_p": float(id_orig_p),
            "id_resp_p": float(id_resp_p),
            "duration": float(duration),
            "orig_bytes": float(orig_bytes),
            "resp_bytes": float(resp_bytes),
            "missed_bytes": float(missed_bytes),
            "orig_pkts": float(orig_pkts),
            "orig_ip_bytes": float(orig_ip_bytes),
            "resp_pkts": float(resp_pkts),
            "resp_ip_bytes": float(resp_ip_bytes),
            "clean_label": clean_labels[i],
            INDEXED_TARGET_COLUMN: int(cls_idx)
        })

    pdf = pd.DataFrame(data)
    df = spark.createDataFrame(pdf)
    logger.info(f"Synthetic data generation completed ({df.count():,} rows).")
    return df
