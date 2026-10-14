"""
Configuration Module for IoT Intrusion Detection System (PySpark)
Defines constants, directory paths, schema definitions, label mappings,
and official benchmark metrics from:
Ahmed Alrefaei & Mohammad Ilyas (2024), Sensors (MDPI), 24(14): 4516.
DOI: 10.3390/s24144516
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_DATA_FILE = DATA_DIR / "iot23_combined.csv"
METRICS_CSV_FILE = RESULTS_DIR / "metrics_summary.csv"
BENCHMARK_COMPARISON_CSV = RESULTS_DIR / "benchmark_comparison.csv"
PER_CLASS_METRICS_CSV = RESULTS_DIR / "per_class_metrics.csv"

CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrices.png"
PERFORMANCE_COMPARISON_PNG = RESULTS_DIR / "performance_comparison.png"
LATENCY_TRADEOFF_PNG = RESULTS_DIR / "latency_tradeoff.png"

# Ensure output directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CLASS DEFINITIONS & MAPPINGS (5 CLASSES MULTICLASS CLASSIFICATION)
# ==============================================================================
# Class 0: Benign (Normal IoT traffic)
# Class 1: DDoS (Distributed Denial of Service)
# Class 2: Okiru (Mirai-derived botnet variant)
# Class 3: PartOfAHorizontalPortScan (Horizontal Port Scan reconnaissance)
# Class 4: Attack (Unknown / General attack label in IoT-23)

CLASS_NAMES = [
    "Benign",
    "DDoS",
    "Okiru",
    "PartOfAHorizontalPortScan",
    "Attack"
]

CLASS_TO_INDEX = {
    "Benign": 0,
    "DDoS": 1,
    "Okiru": 2,
    "PartOfAHorizontalPortScan": 3,
    "Attack": 4
}

INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}

# Mapping from raw IoT-23 string variations to standardized class names
LABEL_STANDARDIZATION_MAP = {
    # Benign variations
    "benign": "Benign",
    "-   benign   -": "Benign",
    
    # DDoS variations
    "ddos": "DDoS",
    "c&c-ddos": "DDoS",
    
    # Okiru variations
    "okiru": "Okiru",
    "okiru-attack": "Okiru",
    
    # Horizontal Port Scan variations
    "partofahorizontalportscan": "PartOfAHorizontalPortScan",
    "part of a horizontal port scan": "PartOfAHorizontalPortScan",
    "horizontalportscan": "PartOfAHorizontalPortScan",
    "horizontal-port-scan": "PartOfAHorizontalPortScan",
    "partofahorizontalportscan-attack": "PartOfAHorizontalPortScan",
    
    # Attack (referred to as 'Unknown attack' in the paper)
    "attack": "Attack",
    "c&c": "Attack",
    "c&c-heartbeat": "Attack",
    "c&c-heartbeat-attack": "Attack",
    "c&c-filedownload": "Attack",
    "c&c-mirai": "Attack",
    "c&c-torii": "Attack",
    "filedownload": "Attack"
}

# ==============================================================================
# 3. SCHEMA DEFINITIONS & FEATURE GROUPS
# ==============================================================================
# Raw column renames: Zeek logs use dot notation (e.g., id.orig_h),
# which must be sanitized to underscores for PySpark ML compatibility.
COLUMN_RENAME_MAP = {
    "id.orig_h": "id_orig_h",
    "id.orig_p": "id_orig_p",
    "id.resp_h": "id_resp_h",
    "id.resp_p": "id_resp_p"
}

# Identifiers and non-predictive attributes to drop
DROP_COLUMNS = [
    "Unnamed: 0",
    "ts",
    "uid",
    "id_orig_h",
    "id_resp_h",
    "local_orig",
    "local_resp"
]

# Categorical attributes requiring StringIndexer + OneHotEncoder
CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "conn_state",
    "history"
]

# Numerical attributes requiring imputation, VectorAssembler, & MinMaxScaler
NUMERICAL_COLUMNS = [
    "id_orig_p",
    "id_resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes"
]

TARGET_COLUMN = "label"
INDEXED_TARGET_COLUMN = "label_idx"
FEATURES_VECTOR_COL = "raw_features"
SCALED_FEATURES_COL = "scaled_features"
SELECTED_FEATURES_COL = "features"

# ==============================================================================
# 4. DEFAULT MODEL HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2
DEFAULT_K_BEST = 12

MODEL_CONFIGS = {
    "lr": {
        "name": "OneVsRest Logistic Regression",
        "regParam": 0.01,
        "maxIter": 100,
        "family": "binomial"
    },
    "rf": {
        "name": "Random Forest Classifier",
        "numTrees": 50,
        "maxDepth": 10,
        "seed": RANDOM_SEED
    },
    "dt": {
        "name": "OneVsRest Decision Tree",
        "maxDepth": 10,
        "seed": RANDOM_SEED
    },
    "gbt": {
        "name": "Gradient-Boosted Trees (XGBoost Equivalent)",
        "maxIter": 30,
        "maxDepth": 8,
        "seed": RANDOM_SEED
    }
}

# ==============================================================================
# 5. OFFICIAL PAPER BENCHMARK METRICS (Alrefaei & Ilyas, 2024)
# ==============================================================================
# Sourced directly from Tables 2 to 11 in Sensors 2024, 24(14), 4516
PAPER_METADATA = {
    "title": "Using Machine Learning Multiclass Classification Technique to Detect IoT Attacks in Real Time",
    "authors": "Ahmed Alrefaei & Mohammad Ilyas",
    "journal": "Sensors (MDPI)",
    "volume": 24,
    "issue": 14,
    "article_number": 4516,
    "year": 2024,
    "doi": "https://doi.org/10.3390/s24144516",
    "dataset": "IoT-23 Malware Types (Combined)",
    "total_records": 1444674
}

BENCHMARK_OVERALL = {
    "OneVsRest Logistic Regression": {
        "accuracy": 0.8298,
        "cv_accuracy_5fold": 0.8032,
        "training_time_s": 95.6417,
        "prediction_time_s": 0.4987
    },
    "Random Forest Classifier": {
        "accuracy": 0.9854,
        "cv_accuracy_5fold": 0.9832,
        "training_time_s": 30.8600,
        "prediction_time_s": 0.0311
    },
    "OneVsRest Decision Tree": {
        "accuracy": 0.9761,
        "cv_accuracy_5fold": 0.9657,
        "training_time_s": 6.8637,
        "prediction_time_s": 0.0804
    },
    "K-Nearest Neighbors": {
        "accuracy": 0.9887,
        "cv_accuracy_5fold": 0.9855,
        "training_time_s": 10.9469,
        "prediction_time_s": 0.9469
    },
    "Extreme Gradient Boosting (XGB)": {
        "accuracy": 0.9889,
        "cv_accuracy_5fold": 0.9861,
        "training_time_s": 50.3426,
        "prediction_time_s": 0.7381
    }
}

BENCHMARK_PER_CLASS = {
    "OneVsRest Logistic Regression": {
        "Benign": {"precision": 0.9628, "recall": 0.2869, "f1": 0.4421},
        "DDoS": {"precision": 0.9973, "recall": 0.9992, "f1": 0.9983},
        "Okiru": {"precision": 0.8926, "recall": 0.9580, "f1": 0.9242},
        "PartOfAHorizontalPortScan": {"precision": 0.5714, "recall": 0.9086, "f1": 0.7016},
        "Attack": {"precision": 0.9534, "recall": 1.0000, "f1": 0.9762}
    },
    "Random Forest Classifier": {
        "Benign": {"precision": 0.9779, "recall": 0.9551, "f1": 0.9664},
        "DDoS": {"precision": 0.9985, "recall": 1.0000, "f1": 0.9992},
        "Okiru": {"precision": 0.9972, "recall": 0.9964, "f1": 0.9968},
        "PartOfAHorizontalPortScan": {"precision": 0.9547, "recall": 0.9756, "f1": 0.9650},
        "Attack": {"precision": 0.9987, "recall": 0.9998, "f1": 0.9992}
    },
    "OneVsRest Decision Tree": {
        "Benign": {"precision": 0.9740, "recall": 0.9083, "f1": 0.9400},
        "DDoS": {"precision": 0.9986, "recall": 1.0000, "f1": 0.9993},
        "Okiru": {"precision": 0.9962, "recall": 1.0000, "f1": 0.9981},
        "PartOfAHorizontalPortScan": {"precision": 0.9151, "recall": 0.9725, "f1": 0.9429},
        "Attack": {"precision": 0.9987, "recall": 0.9996, "f1": 0.9991}
    },
    "Extreme Gradient Boosting (XGB)": {
        "Benign": {"precision": 0.9805, "recall": 0.9637, "f1": 0.9720},
        "DDoS": {"precision": 0.9996, "recall": 1.0000, "f1": 0.9998},
        "Okiru": {"precision": 0.9996, "recall": 0.9996, "f1": 0.9990},
        "PartOfAHorizontalPortScan": {"precision": 0.9670, "recall": 0.9812, "f1": 0.9741},
        "Attack": {"precision": 0.9996, "recall": 1.0000, "f1": 0.9998}
    }
}
