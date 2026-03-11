"""
Comprehensive Multi-Metric Analysis & Performance Benchmarking
==============================================================

Generates 6 publication-quality visualisations for the Spam Call
Classification system and saves them to  analysis_output/.

Charts produced:
  1. Confusion Matrix            (confusion_matrix.png)
  2. ROC Curve                   (roc_curve.png)
  3. Multi-Metric Bar Comparison (multi_metric_comparison.png)
  4. Radar / Spider Chart        (radar_chart.png)
  5. Benchmark Heatmap           (benchmark_heatmap.png)
  6. Precision–Recall Trade-off  (precision_recall_tradeoff.png)

Usage:
    python generate_analysis.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless, no GUI needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ─── project imports ───
from classifier import ONNXClassifier

# ─── paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
METRICS_JSON = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── global style ───
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#181828",
    "axes.edgecolor":   "#3A3A5C",
    "axes.labelcolor":  "#E0E0F0",
    "text.color":       "#E0E0F0",
    "xtick.color":      "#B0B0D0",
    "ytick.color":      "#B0B0D0",
    "grid.color":       "#2A2A48",
    "grid.alpha":        0.6,
    "savefig.facecolor":"#0F0F1A",
    "savefig.edgecolor":"#0F0F1A",
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Segoe UI", "DejaVu Sans", "Arial"],
})

# ─── colour palette ───
ACCENT       = "#7C4DFF"   # electric purple
ACCENT2      = "#00E5FF"   # cyan
ACCENT3      = "#FF6D00"   # orange
ACCENT4      = "#00E676"   # green
GRAD_CMAP    = mcolors.LinearSegmentedColormap.from_list(
    "brand", ["#0F0F1A", "#311B92", "#7C4DFF", "#B388FF"]
)
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "heat_brand", ["#181828", "#1A237E", "#304FFE", "#448AFF", "#80D8FF", "#B2FF59"]
)

# ============================================================
# 1.  TEST DATA  (same set used in test_model_metrics.py)
# ============================================================
TEST_DATA = [
    {"text": "Congratulations! You've won a free prize. Call now to claim!", "label": "Spam"},
    {"text": "Hey, are you coming to the meeting tomorrow?", "label": "Ham"},
    {"text": "Sir, bank loan offer irukku, 2% interest la kedaikum.", "label": "Spam"},
    {"text": "Naan late varuvena, 15 minutes la varuven.", "label": "Ham"},
    {"text": "Sir our number is OTP sir, OTP number, so we can enter the OTP and enter the address.", "label": "Spam"},
    {"text": "Bro, cricket match paakka pogalama?", "label": "Ham"},
    {"text": "Sir claim pannunga sir, ungalukku special offer irukku.", "label": "Spam"},
    {"text": "Your bank account has been compromised. Verify immediately!", "label": "Spam"},
    {"text": "Can you please send me the presentation file?", "label": "Ham"},
    {"text": "Enakku inniki konjam udambu mudiyala, office vara maaten.", "label": "Ham"},
    {"text": "Hello madam, neenga select aagitinga 1 lakh prize ku. Details anupunga.", "label": "Spam"},
    {"text": "Aadhaar card suspend agirchu, update panna indha link click pannunga.", "label": "Spam"},
    {"text": "Dei evening yenga polam?", "label": "Ham"},
    {"text": "Amazon customer support calling. Your package is delayed.", "label": "Ham"},
    {"text": "Urgent! Your credit card is blocked due to suspicious activity. Call this number.", "label": "Spam"},
    {"text": "Hi sir, can I get your email ID to send the invoice?", "label": "Ham"},
]


def _run_inference(classifier):
    """Run the MuRIL hybrid classifier on all test samples.

    Returns
    -------
    y_true  : list[int]  — ground-truth  (0=Ham, 1=Spam)
    y_pred  : list[int]  — predicted     (0=Ham, 1=Spam)
    y_scores: list[float] — spam probability (continuous)
    """
    y_true, y_pred, y_scores = [], [], []
    for item in TEST_DATA:
        true_label = 1 if item["label"] == "Spam" else 0
        result = classifier.classify(item["text"])
        pred_label = 1 if result["label"] == "Spam" else 0
        spam_prob  = result["probabilities"]["Spam"]

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(spam_prob)
    return y_true, y_pred, y_scores


# ============================================================
# CHART 1 — Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Ham (Legit)", "Spam (Threat)"]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=GRAD_CMAP,
        xticklabels=labels, yticklabels=labels,
        linewidths=2, linecolor="#0F0F1A",
        annot_kws={"size": 28, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label",      fontsize=14, fontweight="bold", labelpad=12)
    ax.set_title("Confusion Matrix — Fine-Tuned MuRIL Hybrid Classifier",
                 fontsize=16, fontweight="bold", pad=18, color=ACCENT2)
    # accuracy badge
    acc = accuracy_score(y_true, y_pred)
    ax.text(1.0, -0.12, f"Accuracy: {acc*100:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=13,
            color=ACCENT4, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# CHART 2 — ROC Curve
# ============================================================
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(9, 7))

    # filled area under curve
    ax.fill_between(fpr, tpr, alpha=0.18, color=ACCENT)
    ax.plot(fpr, tpr, color=ACCENT, lw=3,
            label=f"MuRIL Hybrid  (AUC = {roc_auc:.3f})")

    # diagonal reference
    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color="#555577",
            label="Random Guess (AUC = 0.500)")

    # markers on actual points
    ax.scatter(fpr, tpr, color=ACCENT2, s=50, zorder=5, edgecolors="#0F0F1A", lw=1.2)

    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontsize=14, fontweight="bold")
    ax.set_title("ROC Curve — Spam vs Ham Classification",
                 fontsize=16, fontweight="bold", pad=18, color=ACCENT2)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.8,
              facecolor="#181828", edgecolor="#3A3A5C")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# CHART 3 — Multi-Metric Bar Comparison (all models)
# ============================================================
def plot_multi_metric_comparison(all_metrics):
    models  = list(all_metrics.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = [ACCENT, ACCENT2, ACCENT3, ACCENT4]

    n_models  = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(18, 9))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [all_metrics[m][metric] for m in models]
        offset = (i - n_metrics / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, values, bar_w, label=metric,
                       color=color, edgecolor="#0F0F1A", lw=0.6,
                       alpha=0.92, zorder=3)
        # value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=38, ha="right", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("Multi-Metric Performance Comparison Across All Models",
                 fontsize=17, fontweight="bold", pad=20, color=ACCENT2)
    ax.legend(fontsize=12, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              framealpha=0.8, facecolor="#181828", edgecolor="#3A3A5C")

    # Highlight the fine-tuned model tick label
    tick_labels = ax.get_xticklabels()
    tick_labels[-1].set_fontweight("bold")
    tick_labels[-1].set_color(ACCENT2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "multi_metric_comparison.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# CHART 4 — Radar / Spider Chart (top models)
# ============================================================
def plot_radar_chart(all_metrics):
    # Pick top 4 by F1 and always include our model
    sorted_models = sorted(all_metrics.items(), key=lambda kv: kv[1]["F1-Score"], reverse=True)
    top_models = []
    for name, _ in sorted_models:
        if len(top_models) >= 4:
            break
        top_models.append(name)
    if "Fine-Tuned MuRIL (Ours)" not in top_models:
        top_models[-1] = "Fine-Tuned MuRIL (Ours)"

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    ax.set_facecolor("#181828")
    fig.patch.set_facecolor("#0F0F1A")

    for i, model in enumerate(top_models):
        vals = [all_metrics[model][m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2.5, color=colors[i], label=model, markersize=7)
        ax.fill(angles, vals, alpha=0.12, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=9, color="#8888AA")
    ax.yaxis.grid(True, color="#2A2A48", lw=0.8)
    ax.xaxis.grid(True, color="#2A2A48", lw=0.8)
    ax.spines["polar"].set_color("#3A3A5C")

    ax.set_title("Radar Chart — Top Models Comparison",
                 fontsize=16, fontweight="bold", pad=30, color=ACCENT2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=11, framealpha=0.8,
              facecolor="#181828", edgecolor="#3A3A5C")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "radar_chart.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# CHART 5 — Benchmark Heatmap
# ============================================================
def plot_benchmark_heatmap(all_metrics):
    models  = list(all_metrics.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data = np.array([[all_metrics[m][met] for met in metrics] for m in models])

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap=HEATMAP_CMAP,
        xticklabels=metrics, yticklabels=models,
        linewidths=2, linecolor="#0F0F1A",
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"shrink": 0.85, "label": "Score (%)"},
        vmin=30, vmax=100,
        ax=ax,
    )
    ax.set_title("Performance Benchmark Heatmap — All Models",
                 fontsize=17, fontweight="bold", pad=18, color=ACCENT2)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    # Highlight our model row
    y_idx = models.index("Fine-Tuned MuRIL (Ours)")
    ax.get_yticklabels()[y_idx].set_fontweight("bold")
    ax.get_yticklabels()[y_idx].set_color(ACCENT2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "benchmark_heatmap.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# CHART 6 — Precision-Recall Trade-off Scatter
# ============================================================
def plot_precision_recall_tradeoff(all_metrics):
    models = list(all_metrics.keys())
    prec   = [all_metrics[m]["Precision"] for m in models]
    rec    = [all_metrics[m]["Recall"]    for m in models]
    f1     = [all_metrics[m]["F1-Score"]  for m in models]

    # Bubble size proportional to F1
    sizes = [(s / 100) ** 2 * 1800 + 120 for s in f1]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each model as a bubble
    scatter = ax.scatter(
        rec, prec, s=sizes, c=f1,
        cmap=HEATMAP_CMAP, edgecolors="#E0E0F0", lw=1.5,
        alpha=0.88, zorder=5, vmin=50, vmax=100,
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, label="F1-Score (%)")
    cbar.ax.yaxis.label.set_color("#E0E0F0")

    # Annotate each point
    for i, model in enumerate(models):
        short = model.replace("Fine-Tuned MuRIL (Ours)", "MuRIL\n(Ours)")
        short = short.replace("Ensemble (Soft Voting)", "Ensemble")
        short = short.replace("Logistic Regression", "Logistic\nRegression")
        short = short.replace("Gradient Boosting", "Gradient\nBoosting")
        short = short.replace("Random Forest", "Random\nForest")
        short = short.replace("Decision Tree", "Decision\nTree")

        color = ACCENT2 if "Ours" in model else "#C0C0E0"
        weight = "bold" if "Ours" in model else "normal"
        ax.annotate(
            short, (rec[i], prec[i]),
            textcoords="offset points", xytext=(0, 18),
            ha="center", fontsize=9, color=color, fontweight=weight,
        )

    # Iso-F1 curves
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r = np.linspace(0.01, 1.0, 200)
        p = (f1_val * r) / (2 * r - f1_val)
        mask = (p > 0) & (p <= 1.05)
        ax.plot(r[mask] * 100, p[mask] * 100, ls="--", lw=0.9,
                color="#555577", alpha=0.5)
        # label at start of the curve
        valid_idx = np.where(mask)[0]
        if len(valid_idx) > 0:
            ax.text(r[valid_idx[-1]] * 100 + 1, p[valid_idx[-1]] * 100,
                    f"F1={f1_val}", fontsize=8, color="#666688", alpha=0.7)

    ax.set_xlabel("Recall (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Precision (%)", fontsize=14, fontweight="bold")
    ax.set_title("Precision–Recall Trade-off (bubble = F1 magnitude)",
                 fontsize=16, fontweight="bold", pad=18, color=ACCENT2)
    ax.set_xlim(25, 108)
    ax.set_ylim(50, 108)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "precision_recall_tradeoff.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  Multi-Metric Analysis & Performance Benchmarking")
    print("=" * 65)

    # --- Load classifier & run inference ---
    print("\n[1/3] Loading fine-tuned MuRIL classifier …")
    classifier = ONNXClassifier()
    try:
        classifier.load()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)

    print("[2/3] Running inference on test set …")
    y_true, y_pred, y_scores = _run_inference(classifier)

    # Quick console report
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100
    print(f"\n  Fine-Tuned MuRIL Results:")
    print(f"    Accuracy  : {acc:.1f}%")
    print(f"    Precision : {prec:.1f}%")
    print(f"    Recall    : {rec:.1f}%")
    print(f"    F1-Score  : {f1:.1f}%")

    muril_metrics = {
        "Accuracy":  round(acc,  2),
        "Precision": round(prec, 2),
        "Recall":    round(rec,  2),
        "F1-Score":  round(f1,   2),
    }

    # --- Load baseline metrics ---
    with open(METRICS_JSON, "r") as fh:
        baseline_metrics = json.load(fh)

    all_metrics = dict(baseline_metrics)
    all_metrics["Fine-Tuned MuRIL (Ours)"] = muril_metrics

    # --- Generate all charts ---
    print(f"\n[3/3] Generating charts → {OUTPUT_DIR}")
    print("-" * 50)

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)
    plot_multi_metric_comparison(all_metrics)
    plot_radar_chart(all_metrics)
    plot_benchmark_heatmap(all_metrics)
    plot_precision_recall_tradeoff(all_metrics)

    print("-" * 50)
    print(f"\n  ✅  All 6 charts saved to: {OUTPUT_DIR}")
    print(f"  📊  Classification Report:\n")
    print(classification_report(
        y_true, y_pred,
        target_names=["Ham (Legit)", "Spam (Threat)"],
        digits=3
    ))


if __name__ == "__main__":
    main()
