"""
Evaluation & Visualization Module for IoT Intrusion Detection System
Implements:
1. Overall multiclass metrics (Accuracy, Weighted Precision, Recall, F1).
2. Per-class metrics (Equations 1-4 in paper: Precision, Recall, F1-Score).
3. 5x5 Confusion Matrix generation.
4. Direct comparison table against Alrefaei & Ilyas (2024) benchmark results.
5. Export of 3 high-resolution publication-quality PNG charts into results/.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
except ImportError:
    DataFrame = None
    F = None
    MulticlassClassificationEvaluator = None

from src.config import (
    CLASS_NAMES,
    INDEX_TO_CLASS,
    INDEXED_TARGET_COLUMN,
    BENCHMARK_OVERALL,
    BENCHMARK_PER_CLASS,
    METRICS_CSV_FILE,
    BENCHMARK_COMPARISON_CSV,
    PER_CLASS_METRICS_CSV,
    CONFUSION_MATRIX_PNG,
    PERFORMANCE_COMPARISON_PNG,
    LATENCY_TRADEOFF_PNG
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure clean aesthetic matplotlib styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_metrics(
    predictions_df: DataFrame,
    model_name: str,
    training_time: float,
    prediction_time: float,
    cv_accuracy: Optional[float] = None,
    label_col: str = INDEXED_TARGET_COLUMN,
    prediction_col: str = "prediction"
) -> Dict[str, Any]:
    """
    Computes overall and class-specific classification metrics from PySpark predictions.

    Args:
        predictions_df: DataFrame containing actual and predicted labels.
        model_name: Readable model display name.
        training_time: Training wall-clock time in seconds.
        prediction_time: Inference wall-clock time in seconds.
        cv_accuracy: Optional cross-validation mean accuracy.
        label_col: Actual label index column name.
        prediction_col: Predicted label column name.

    Returns:
        Dictionary of metrics, per-class metrics, and confusion matrix.
    """
    logger.info(f"Computing evaluation metrics for: {model_name}...")

    # PySpark Multiclass Evaluators
    acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    prec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="weightedRecall")
    f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="weightedFMeasure")

    accuracy = float(acc_eval.evaluate(predictions_df))
    precision = float(prec_eval.evaluate(predictions_df))
    recall = float(rec_eval.evaluate(predictions_df))
    f1_score = float(f1_eval.evaluate(predictions_df))

    # Pull labels to driver for detailed confusion matrix and per-class metrics
    logger.info("Extracting prediction pairs for confusion matrix...")
    pred_pairs = predictions_df.select(label_col, prediction_col).collect()
    y_true = np.array([int(r[0]) for r in pred_pairs])
    y_pred = np.array([int(r[1]) for r in pred_pairs])

    num_classes = len(CLASS_NAMES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_matrix[t, p] += 1

    # Calculate per-class metrics (Precision, Recall, F1) matching Equations 1-4
    per_class_metrics = {}
    for i, c_name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        c_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        c_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        c_f1 = (2 * c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

        per_class_metrics[c_name] = {
            "precision": float(c_prec),
            "recall": float(c_rec),
            "f1": float(c_f1),
            "support": int(conf_matrix[i, :].sum())
        }

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1_score,
        "cv_accuracy_5fold": cv_accuracy,
        "training_time_s": training_time,
        "prediction_time_s": prediction_time,
        "per_class": per_class_metrics,
        "confusion_matrix": conf_matrix
    }


def export_and_display_results(
    all_results: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prints structured tables to the terminal and exports CSV summaries comparing
    experimental results against the paper benchmark.

    Args:
        all_results: List of model metric dictionaries.

    Returns:
        (summary_df, comparison_df)
    """
    logger.info("Exporting results and generating benchmark comparisons...")

    # 1. Summary DataFrame
    summary_rows = []
    for res in all_results:
        summary_rows.append({
            "Model": res["model_name"],
            "Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Weighted Prec (%)": round(res["weighted_precision"] * 100, 2),
            "Weighted Rec (%)": round(res["weighted_recall"] * 100, 2),
            "Weighted F1 (%)": round(res["weighted_f1"] * 100, 2),
            "5-Fold CV Acc (%)": round(res["cv_accuracy_5fold"] * 100, 2) if res.get("cv_accuracy_5fold") else "N/A",
            "Training Time (s)": round(res["training_time_s"], 4),
            "Prediction Latency (s)": round(res["prediction_time_s"], 4)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(METRICS_CSV_FILE, index=False)
    logger.info(f"Metrics summary saved to: {METRICS_CSV_FILE}")

    # 2. Per-Class DataFrame
    per_class_rows = []
    for res in all_results:
        m_name = res["model_name"]
        for c_name, c_vals in res["per_class"].items():
            per_class_rows.append({
                "Model": m_name,
                "Class": c_name,
                "Support": c_vals["support"],
                "Precision (%)": round(c_vals["precision"] * 100, 2),
                "Recall (%)": round(c_vals["recall"] * 100, 2),
                "F1 Score (%)": round(c_vals["f1"] * 100, 2)
            })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(PER_CLASS_METRICS_CSV, index=False)
    logger.info(f"Per-class metrics saved to: {PER_CLASS_METRICS_CSV}")

    # 3. Benchmark Comparison DataFrame
    comp_rows = []
    for res in all_results:
        m_name = res["model_name"]
        matched_bench_key = None
        for b_key in BENCHMARK_OVERALL.keys():
            if (b_key.lower() in m_name.lower()) or (m_name.lower() in b_key.lower()) or \
               ("forest" in m_name.lower() and "forest" in b_key.lower()) or \
               ("decision" in m_name.lower() and "decision" in b_key.lower()) or \
               ("logistic" in m_name.lower() and "logistic" in b_key.lower()) or \
               ("gradient" in m_name.lower() and "gradient" in b_key.lower()) or \
               ("xgb" in m_name.lower() and "xgb" in b_key.lower()):
                matched_bench_key = b_key
                break

        if matched_bench_key:
            bench_vals = BENCHMARK_OVERALL[matched_bench_key]
            acc_ours = res["accuracy"] * 100
            acc_paper = bench_vals["accuracy"] * 100
            delta_acc = acc_ours - acc_paper

            t_train_ours = res["training_time_s"]
            t_train_paper = bench_vals["training_time_s"]

            t_pred_ours = res["prediction_time_s"]
            t_pred_paper = bench_vals["prediction_time_s"]

            comp_rows.append({
                "Model": m_name,
                "Accuracy (Ours %)": round(acc_ours, 2),
                "Accuracy (Paper %)": round(acc_paper, 2),
                "Delta Acc (%)": round(delta_acc, 2),
                "Train Time (Ours s)": round(t_train_ours, 3),
                "Train Time (Paper s)": round(t_train_paper, 3),
                "Pred Latency (Ours s)": round(t_pred_ours, 4),
                "Pred Latency (Paper s)": round(t_pred_paper, 4),
                "Replication Status": "PASSED (Consistent)" if abs(delta_acc) <= 6.0 else "DEVIATED"
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(BENCHMARK_COMPARISON_CSV, index=False)
    logger.info(f"Benchmark comparison saved to: {BENCHMARK_COMPARISON_CSV}")

    # Print Clean Formatted Console Table
    print("\n" + "=" * 90)
    print("           EXPERIMENTAL RESULTS SUMMARY (Apache Spark PySpark MLlib)")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 90)
    print("      REPLICATION BENCHMARK COMPARISON (vs Alrefaei & Ilyas, 2024)")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df, comp_df


def plot_confusion_matrices(
    all_results: List[Dict[str, Any]],
    output_path: Path = CONFUSION_MATRIX_PNG
):
    """
    Plots high-resolution multi-panel heatmaps of confusion matrices for all evaluated models.
    """
    n_models = len(all_results)
    if n_models == 0:
        return

    cols = 2
    rows = (n_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    # Abbreviated labels for clear heatmap display
    short_labels = ["Benign", "DDoS", "Okiru", "PortScan", "Attack"]

    for i, res in enumerate(all_results):
        ax = axes[i]
        cm = res["confusion_matrix"]
        
        # Calculate percentage representation
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_perc[r, c]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            annot_kws={"size": 9}
        )
        ax.set_title(f"{res['model_name']}\nAccuracy: {res['accuracy'] * 100:.2f}%", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Class", fontsize=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Multiclass Confusion Matrices across Evaluated Models (IoT-23)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices plot saved to: {output_path}")


def plot_performance_comparison(
    all_results: List[Dict[str, Any]],
    output_path: Path = PERFORMANCE_COMPARISON_PNG
):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1
    across models against the official paper benchmark.
    """
    if not all_results:
        return

    model_names = [r["model_name"] for r in all_results]
    accuracies = [r["accuracy"] * 100 for r in all_results]
    precisions = [r["weighted_precision"] * 100 for r in all_results]
    recalls = [r["weighted_recall"] * 100 for r in all_results]
    f1s = [r["weighted_f1"] * 100 for r in all_results]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    
    rects1 = ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy", color="#1f77b4")
    rects2 = ax.bar(x - 0.5 * width, precisions, width, label="Weighted Precision", color="#2ca02c")
    rects3 = ax.bar(x + 0.5 * width, recalls, width, label="Weighted Recall", color="#ff7f0e")
    rects4 = ax.bar(x + 1.5 * width, f1s, width, label="Weighted F1-Score", color="#9467bd")

    # Add reference benchmark horizontal lines for Top Paper Performers
    ax.axhline(98.89, color="#d62728", linestyle="--", linewidth=1.2, label="Paper Peak Accuracy (XGB: 98.89%)")
    ax.axhline(98.54, color="#e377c2", linestyle=":", linewidth=1.2, label="Paper RF Accuracy (98.54%)")

    display_names = []
    for m in model_names:
        if "Logistic" in m:
            display_names.append("Logistic Regression\n(OVR)")
        elif "Random Forest" in m:
            display_names.append("Random Forest")
        elif "Decision Tree" in m:
            display_names.append("Decision Tree\n(OVR)")
        elif "Gradient" in m or "GBT" in m:
            display_names.append("Gradient-Boosted\nTrees (GBT / XGB)")
        else:
            display_names.append(m)

    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison across Models vs Alrefaei & Ilyas (2024) Benchmark", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="bold")
    ax.set_ylim(70, 103)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Performance comparison plot saved to: {output_path}")


def plot_latency_tradeoff(
    all_results: List[Dict[str, Any]],
    output_path: Path = LATENCY_TRADEOFF_PNG
):
    """
    Generates a polished dual-panel publication trade-off chart visualizing:
    1. Prediction Latency (seconds) vs Accuracy (Real-time IoT Edge viability).
       Features clean rounded callouts with pointer arrows and zero label collisions.
    2. Training Time (seconds) across models with multiline bold ticks and headroom.
    Highlights Random Forest's rapid 0.0311s inference time demonstrated in the paper.
    """
    if not all_results:
        return

    def clean_model_name(raw_name: str) -> str:
        if "Logistic" in raw_name:
            return "Logistic Regression (OVR)"
        elif "Random Forest" in raw_name:
            return "Random Forest"
        elif "Decision Tree" in raw_name:
            return "Decision Tree (OVR)"
        elif "Gradient" in raw_name or "GBT" in raw_name:
            return "GBT (XGBoost Eq.)"
        return raw_name

    def multiline_label(raw_name: str) -> str:
        if "Logistic" in raw_name:
            return "Logistic Regression\n(OVR)"
        elif "Random Forest" in raw_name:
            return "Random Forest"
        elif "Decision Tree" in raw_name:
            return "Decision Tree\n(OVR)"
        elif "Gradient" in raw_name or "GBT" in raw_name:
            return "Gradient-Boosted\nTrees (GBT)"
        return raw_name

    short_names = [clean_model_name(r["model_name"]) for r in all_results]
    bar_names = [multiline_label(r["model_name"]) for r in all_results]
    pred_times = [r["prediction_time_s"] for r in all_results]
    train_times = [r["training_time_s"] for r in all_results]
    accuracies = [r["accuracy"] * 100 for r in all_results]

    palette = ["#2b5c8f", "#2e7d32", "#d97706", "#c0392b"][:len(short_names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel 1: Prediction Latency vs Accuracy Scatter
    for i, name in enumerate(short_names):
        ax1.scatter(
            pred_times[i], accuracies[i],
            s=220, color=palette[i],
            label=name, edgecolors="#1a1a1a", linewidth=1.5, zorder=5
        )

    # Reference point for paper RF benchmark
    ax1.scatter(
        0.0311, 98.54,
        s=250, color="#8e44ad", marker="*",
        label="Paper RF (0.0311s, 98.54%)",
        edgecolors="#1a1a1a", linewidth=1.2, zorder=6
    )

    # Non-overlapping callout placement: (dx, dy, ha, va, rad)
    annotation_configs = {
        "Logistic Regression (OVR)": (-25, 25, "right", "bottom", -0.15),
        "Random Forest": (-25, -28, "right", "top", 0.15),
        "Decision Tree (OVR)": (25, -28, "left", "top", -0.15),
        "GBT (XGBoost Eq.)": (25, 25, "left", "bottom", 0.15),
    }

    for i, name in enumerate(short_names):
        dx, dy, ha, va, rad = annotation_configs.get(name, (20, 10, "left", "center", 0.0))
        ax1.annotate(
            f"{name}\n({pred_times[i]:.4f}s, {accuracies[i]:.1f}%)",
            xy=(pred_times[i], accuracies[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fefefe", edgecolor="#b0bec5", alpha=0.92, lw=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={rad}", color="#455a64", lw=1.1)
        )

    # Annotate Paper RF Reference Star
    ax1.annotate(
        "Paper RF Benchmark\n(0.0311s, 98.54%)",
        xy=(0.0311, 98.54),
        xytext=(25, -22),
        textcoords="offset points",
        ha="left", va="top",
        fontsize=9,
        fontweight="bold",
        color="#4a148c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdfbf7", edgecolor="#ba68c8", alpha=0.92, lw=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#8e44ad", lw=1.1)
    )

    ax1.set_xlabel("Prediction Latency (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overall Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Real-Time Viability: Latency vs Detection Accuracy", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xscale("log")
    ax1.set_xlim(0.015, 20)
    ax1.set_ylim(80, 102)
    ax1.set_yticks([80, 85, 90, 95, 100])
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.92, edgecolor="#cccccc")

    # Panel 2: Training Time Bar Chart
    bars = ax2.bar(range(len(bar_names)), train_times, color=palette, edgecolor="#1a1a1a", linewidth=1.2, width=0.55)
    ax2.set_ylabel("Training Time (Seconds)", fontsize=11, fontweight="bold")
    ax2.set_title("Computational Efficiency: Model Training Wall-Clock Time", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(range(len(bar_names)))
    ax2.set_xticklabels(bar_names, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(train_times) * 1.18)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )

    plt.tight_layout(pad=2.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Latency trade-off plot saved to: {output_path}")


def generate_all_visualizations(all_results: List[Dict[str, Any]]):
    """
    Generates all 3 required high-resolution plots.
    """
    plot_confusion_matrices(all_results)
    plot_performance_comparison(all_results)
    plot_latency_tradeoff(all_results)
