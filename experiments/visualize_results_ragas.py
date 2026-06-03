#!/usr/bin/env python3
"""
Evaluation Results Visualizer & Table Generator for RAG System

This script reads the evaluation results JSON file (as produced by the
run_ragas_rag_evaluation.py script) and generates:
- Various plots (bar, radar, scatter, boxplots, etc.)
- Summary tables (CSV/Excel, console markdown)

Usage:
    python visualize_results.py --input evaluation_results.json --output_dir ./plots
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


def load_data(input_path: str) -> pd.DataFrame:
    """Load JSON evaluation results into a pandas DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Ensure numeric columns are float
    numeric_cols = [
        "retrieval_recall",
        "retrieval_precision",
        "keyword_hit_rate",
        "context_recall",
        "faithfulness",
        "answer_correctness",
        "rag_elapsed_seconds",
        "retrieval_hit_count",
        "turn",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_dimension_averages(df: pd.DataFrame, output_dir: str):
    """Bar chart of average metrics per dimension."""
    metrics = [
        "retrieval_recall",
        "retrieval_precision",
        "keyword_hit_rate",
        "context_recall",
        "faithfulness",
        "answer_correctness",
    ]
    avg_by_dim = df.groupby("dimension")[metrics].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(avg_by_dim["dimension"]))
    width = 0.12
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            avg_by_dim[metric],
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
        )

    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Average Retrieval & Generation Metrics by Dimension", fontsize=14)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(avg_by_dim["dimension"], rotation=45, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dimension_averages.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/dimension_averages.png")


def plot_radar_chart(df: pd.DataFrame, output_dir: str):
    """Radar chart comparing dimensions across key metrics."""
    metrics = [
        "retrieval_recall",
        "retrieval_precision",
        "keyword_hit_rate",
        "context_recall",
        "faithfulness",
        "answer_correctness",
    ]
    dims = df["dimension"].unique()
    avg_by_dim = df.groupby("dimension")[metrics].mean()

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))

    for dim, color in zip(dims, colors):
        values = avg_by_dim.loc[dim, metrics].values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=dim, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart of Metrics by Dimension", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/radar_chart.png")


def plot_precision_vs_correctness(df: pd.DataFrame, output_dir: str):
    """Scatter plot: retrieval_precision vs answer_correctness, colored by dimension."""
    plt.figure(figsize=(10, 6))
    dims = df["dimension"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
    for dim, color in zip(dims, colors):
        subset = df[df["dimension"] == dim]
        plt.scatter(
            subset["retrieval_precision"],
            subset["answer_correctness"],
            label=dim,
            alpha=0.6,
            s=60,
            color=color,
        )
    plt.xlabel("Retrieval Precision", fontsize=12)
    plt.ylabel("Answer Correctness (Ragas)", fontsize=12)
    plt.title("Retrieval Precision vs Answer Correctness", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_vs_correctness.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/precision_vs_correctness.png")


def plot_recall_vs_faithfulness(df: pd.DataFrame, output_dir: str):
    """Scatter plot: retrieval_recall vs faithfulness."""
    plt.figure(figsize=(10, 6))
    dims = df["dimension"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
    for dim, color in zip(dims, colors):
        subset = df[df["dimension"] == dim]
        plt.scatter(
            subset["retrieval_recall"],
            subset["faithfulness"],
            label=dim,
            alpha=0.6,
            s=60,
            color=color,
        )
    plt.xlabel("Retrieval Recall", fontsize=12)
    plt.ylabel("Faithfulness", fontsize=12)
    plt.title("Retrieval Recall vs Faithfulness", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_vs_faithfulness.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/recall_vs_faithfulness.png")


def plot_response_analysis(df: pd.DataFrame, output_dir: str):
    """Bar chart: proportion of 'I don't know' responses per dimension."""
    df["is_idk"] = df["response"].str.lower().str.startswith("i don't know")
    idk_rate = df.groupby("dimension")["is_idk"].mean().reset_index()
    idk_rate.columns = ["dimension", "idk_rate"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(idk_rate["dimension"], idk_rate["idk_rate"], color="#e74c3c")
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Proportion of 'I don't know'", fontsize=12)
    plt.title("Rate of Model Refusing to Answer", fontsize=14)
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "idk_rate.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/idk_rate.png")


def plot_retrieval_failure_analysis(df: pd.DataFrame, output_dir: str):
    """Bar chart: proportion of samples where retrieval_recall == 0 per dimension."""
    df["recall_zero"] = (df["retrieval_recall"] == 0).astype(float)
    zero_rate = df.groupby("dimension")["recall_zero"].mean().reset_index()
    zero_rate.columns = ["dimension", "recall_zero_rate"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(zero_rate["dimension"], zero_rate["recall_zero_rate"], color="#3498db")
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Proportion with Recall = 0", fontsize=12)
    plt.title("Retrieval Failure Rate (No Relevant Chunk Found)", fontsize=14)
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retrieval_failure_rate.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/retrieval_failure_rate.png")


def plot_elapsed_time_boxplot(df: pd.DataFrame, output_dir: str):
    """Box plot of elapsed time per dimension."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="dimension", y="rag_elapsed_seconds", data=df, palette="viridis")
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Elapsed Time (seconds)", fontsize=12)
    plt.title("RAG Response Time per Dimension", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elapsed_time_boxplot.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/elapsed_time_boxplot.png")


def plot_keyword_hit_vs_correctness(df: pd.DataFrame, output_dir: str):
    """Scatter plot: keyword_hit_rate vs answer_correctness."""
    plt.figure(figsize=(10, 6))
    dims = df["dimension"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
    for dim, color in zip(dims, colors):
        subset = df[df["dimension"] == dim]
        plt.scatter(
            subset["keyword_hit_rate"],
            subset["answer_correctness"],
            label=dim,
            alpha=0.6,
            s=60,
            color=color,
        )
    plt.xlabel("Keyword Hit Rate", fontsize=12)
    plt.ylabel("Answer Correctness", fontsize=12)
    plt.title("Keyword Hit Rate vs Answer Correctness", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "keyword_vs_correctness.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/keyword_vs_correctness.png")


def plot_tag_performance(df: pd.DataFrame, output_dir: str):
    """Average answer_correctness for the most common tags."""
    df_tags = df.explode("tags")
    tag_perf = df_tags.groupby("tags")["answer_correctness"].agg(["mean", "count"]).reset_index()
    tag_perf = tag_perf[tag_perf["count"] >= 3].sort_values("mean", ascending=False)

    if tag_perf.empty:
        print("Not enough tag data to plot.")
        return

    plt.figure(figsize=(12, 6))
    bars = plt.bar(tag_perf["tags"], tag_perf["mean"], color="#9b59b6")
    plt.xlabel("Tag", fontsize=12)
    plt.ylabel("Average Answer Correctness", fontsize=12)
    plt.title("Answer Correctness by Tag (min 3 samples)", fontsize=14)
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tag_performance.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/tag_performance.png")


# ----------------------- TABLE GENERATION -----------------------
def generate_tables(df: pd.DataFrame, output_dir: str):
    """Generate summary tables (CSV, Excel, and console prints)."""
    # Table 1: Average metrics by dimension
    metrics = [
        "retrieval_recall",
        "retrieval_precision",
        "keyword_hit_rate",
        "context_recall",
        "faithfulness",
        "answer_correctness",
        "rag_elapsed_seconds",
    ]
    dim_avg = df.groupby("dimension")[metrics].mean().round(4).reset_index()
    dim_avg["num_samples"] = df.groupby("dimension").size().values
    # Reorder columns
    dim_avg = dim_avg[["dimension", "num_samples"] + metrics]

    # Table 2: Overall averages
    overall_avg = df[metrics].mean().round(4).to_frame().T
    overall_avg["num_samples"] = len(df)

    # Table 3: "I don't know" rate per dimension
    df["is_idk"] = df["response"].str.lower().str.startswith("i don't know")
    idk_rate = df.groupby("dimension")["is_idk"].mean().round(4).reset_index()
    idk_rate.columns = ["dimension", "idk_rate"]
    idk_rate["num_samples"] = df.groupby("dimension").size().values

    # Table 4: Retrieval failure rate (recall = 0) per dimension
    df["recall_zero"] = (df["retrieval_recall"] == 0).astype(float)
    recall_fail = df.groupby("dimension")["recall_zero"].mean().round(4).reset_index()
    recall_fail.columns = ["dimension", "retrieval_failure_rate"]
    recall_fail["num_samples"] = df.groupby("dimension").size().values

    # Table 5: Average answer correctness by tag (for tags with >=3 samples)
    df_tags = df.explode("tags")
    tag_perf = (
        df_tags.groupby("tags")["answer_correctness"]
        .agg(["mean", "count"])
        .reset_index()
        .round(4)
    )
    tag_perf = tag_perf[tag_perf["count"] >= 3].sort_values("mean", ascending=False)

    # Save CSV files
    dim_avg.to_csv(os.path.join(output_dir, "table_dimension_averages.csv"), index=False)
    overall_avg.to_csv(os.path.join(output_dir, "table_overall_averages.csv"), index=False)
    idk_rate.to_csv(os.path.join(output_dir, "table_idk_rate.csv"), index=False)
    recall_fail.to_csv(os.path.join(output_dir, "table_retrieval_failure.csv"), index=False)
    if not tag_perf.empty:
        tag_perf.to_csv(os.path.join(output_dir, "table_tag_performance.csv"), index=False)

    # Save all tables to a single Excel workbook
    with pd.ExcelWriter(os.path.join(output_dir, "summary_tables.xlsx"), engine="openpyxl") as writer:
        dim_avg.to_excel(writer, sheet_name="By Dimension", index=False)
        overall_avg.to_excel(writer, sheet_name="Overall", index=False)
        idk_rate.to_excel(writer, sheet_name="IDK Rate", index=False)
        recall_fail.to_excel(writer, sheet_name="Retrieval Failure", index=False)
        if not tag_perf.empty:
            tag_perf.to_excel(writer, sheet_name="By Tag", index=False)

    # Print tables to console in markdown format
    print("\n" + "=" * 80)
    print("GENERATED TABLES".center(80))
    print("=" * 80)

    print("\n[1] Average Metrics by Dimension\n")
    print(dim_avg.to_string(index=False))
    print("\n" + "-" * 80)

    print("\n[2] Overall Averages\n")
    print(overall_avg.to_string(index=False))
    print("\n" + "-" * 80)

    print("\n[3] 'I don't know' Rate by Dimension\n")
    print(idk_rate.to_string(index=False))
    print("\n" + "-" * 80)

    print("\n[4] Retrieval Failure Rate (Recall = 0) by Dimension\n")
    print(recall_fail.to_string(index=False))
    print("\n" + "-" * 80)

    if not tag_perf.empty:
        print("\n[5] Answer Correctness by Tag (min 3 samples)\n")
        print(tag_perf.to_string(index=False))
        print("\n" + "-" * 80)

    print(f"\nAll tables saved as CSV and Excel in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables from RAG evaluation results.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to evaluation_results.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./plots", help="Directory to save plots and tables"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input)
    print(f"Loaded {len(df)} samples. Dimensions: {df['dimension'].unique()}")

    # Generate plots
    plot_dimension_averages(df, args.output_dir)
    plot_radar_chart(df, args.output_dir)
    plot_precision_vs_correctness(df, args.output_dir)
    plot_recall_vs_faithfulness(df, args.output_dir)
    plot_response_analysis(df, args.output_dir)
    plot_retrieval_failure_analysis(df, args.output_dir)
    plot_elapsed_time_boxplot(df, args.output_dir)
    plot_keyword_hit_vs_correctness(df, args.output_dir)
    plot_tag_performance(df, args.output_dir)

    # Generate tables
    generate_tables(df, args.output_dir)

    print("\nAll done. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()