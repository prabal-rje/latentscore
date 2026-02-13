"""Generate all paper figures from benchmark + HuggingFace data.

Usage:
    conda run -n latentscore-data python data_work/.experiments/plots/plot_all.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path("data_work/.experiments/eval_assets/clap_200row_final_noprefix")
OUT_DIR = Path("data_work/.experiments/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET_REPO = "guprab/latentscore-data"
SCORED_SPLITS = ["SFT-Train", "SFT-Val", "GRPO", "TEST"]

# Consistent model ordering + colors
MODEL_ORDER = [
    "random",
    "base_untrained",
    "sft_finetuned",
    "opus_4.5",
    "gemini_flash",
    "embedding_lookup",
]
MODEL_LABELS = {
    "random": "Random",
    "base_untrained": "Base (untrained)",
    "sft_finetuned": "SFT Fine-tuned",
    "opus_4.5": "Claude Opus 4.5",
    "gemini_flash": "Gemini 3 Flash",
    "embedding_lookup": "Embedding Lookup",
}
MODEL_COLORS = {
    "random": "#999999",
    "base_untrained": "#c4a882",
    "sft_finetuned": "#e8a838",
    "opus_4.5": "#d4785c",
    "gemini_flash": "#4a90d9",
    "embedding_lookup": "#2eaa4f",
}

ALPHA = 0.7
DPI = 200
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (10, 6)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmark_rows() -> list[dict]:
    path = BENCHMARK_DIR / "benchmark_results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_hf_scored_rows() -> list[dict]:
    rows: list[dict] = []
    for split in SCORED_SPLITS:
        path = hf_hub_download(HF_DATASET_REPO, f"2026-01-26_scored/{split}.jsonl", repo_type="dataset")
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def group_by_model(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("success"):
            groups[r["model"]].append(r)
    return groups


# ---------------------------------------------------------------------------
# Figure 1: Config generation latency histograms per model
# ---------------------------------------------------------------------------


def plot_config_latency(rows: list[dict]) -> None:
    by_model = group_by_model(rows)

    # Exclude random (latency is 0) - not interesting
    models = [m for m in MODEL_ORDER if m in by_model and m != "random"]

    fig, axes = plt.subplots(1, len(models), figsize=(16, 4), sharey=True)
    fig.suptitle("Config Generation Latency by Controller", fontsize=14, fontweight="bold", y=1.02)

    for ax, model in zip(axes, models):
        latencies = [r["config_gen_s"] for r in by_model[model] if r.get("config_gen_s") is not None]
        if not latencies:
            continue

        ax.hist(latencies, bins=30, color=MODEL_COLORS[model], alpha=ALPHA, edgecolor="white", linewidth=0.5)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Seconds", fontsize=9)
        med = statistics.median(latencies)
        ax.axvline(med, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax.text(
            med,
            ax.get_ylim()[1] * 0.92,
            f" med={med:.1f}s",
            fontsize=7,
            ha="left",
            va="top",
        )

    axes[0].set_ylabel("Count", fontsize=10)
    fig.tight_layout()

    out = OUT_DIR / "fig_config_latency_histograms.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Audio synthesis latency histogram (all models overlaid)
# ---------------------------------------------------------------------------


def plot_audio_latency(rows: list[dict]) -> None:
    by_model = group_by_model(rows)

    models = [m for m in MODEL_ORDER if m in by_model]
    fig, axes = plt.subplots(1, len(models), figsize=(18, 4), sharey=True)
    fig.suptitle("Audio Synthesis Latency by Controller", fontsize=14, fontweight="bold", y=1.02)

    for ax, model in zip(axes, models):
        latencies = [r["audio_synth_s"] for r in by_model[model] if r.get("audio_synth_s") is not None]
        if not latencies:
            continue

        ax.hist(latencies, bins=30, color=MODEL_COLORS[model], alpha=ALPHA, edgecolor="white", linewidth=0.5)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Seconds", fontsize=9)
        med = statistics.median(latencies)
        ax.axvline(med, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax.text(
            med,
            ax.get_ylim()[1] * 0.92,
            f" med={med:.2f}s",
            fontsize=7,
            ha="left",
            va="top",
        )

    axes[0].set_ylabel("Count", fontsize=10)
    fig.tight_layout()

    out = OUT_DIR / "fig_audio_synth_latency_histogram.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: CLAP score distribution in the HF dataset (selected best-of-5)
# ---------------------------------------------------------------------------


def plot_hf_clap_distribution(hf_rows: list[dict]) -> None:
    # The "scores_external.clap.final_score" is the CLAP score of the SELECTED best candidate
    selected_scores: list[float] = []
    for r in hf_rows:
        score = r.get("scores_external", {}).get("clap", {}).get("final_score")
        if score is not None:
            selected_scores.append(score)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("CLAP Score Distribution — HuggingFace Dataset (Best-of-5 Selected)", fontsize=13, fontweight="bold")

    ax.hist(selected_scores, bins=80, color="#2eaa4f", alpha=ALPHA, edgecolor="white", linewidth=0.5)
    mean_val = statistics.mean(selected_scores)
    med_val = statistics.median(selected_scores)
    ax.axvline(mean_val, color="red", linestyle="-", linewidth=1.5, label=f"mean={mean_val:.4f}")
    ax.axvline(med_val, color="black", linestyle="--", linewidth=1.5, label=f"median={med_val:.4f}")
    ax.set_xlabel("CLAP Final Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"n={len(selected_scores)} vibes across all splits", fontsize=10, style="italic")
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / "fig_hf_clap_selected_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Best-of-N CLAP — selected vs. each candidate position
# ---------------------------------------------------------------------------


def plot_best_of_n_candidates(hf_rows: list[dict]) -> None:
    # Collect rows that have exactly 5 candidate CLAP scores
    candidate_matrix: list[list[float]] = []
    selected_scores: list[float] = []

    for r in hf_rows:
        cand_scores = r.get("candidate_scores", {}).get("clap", [])
        # Filter to rows with exactly 5 non-None scores
        valid = [s for s in cand_scores if s is not None]
        if len(valid) != 5:
            continue

        ext_score = r.get("scores_external", {}).get("clap", {}).get("final_score")
        if ext_score is None:
            continue

        candidate_matrix.append(valid)
        selected_scores.append(ext_score)

    n_rows = len(candidate_matrix)
    arr = np.array(candidate_matrix)  # shape: (n_rows, 5)

    # --- Panel A: Overlaid histograms of each candidate position + selected ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Best-of-5 CLAP Score Analysis — {n_rows:,} vibes from HuggingFace Dataset",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    ax1 = axes[0]
    bins = np.linspace(-0.4, 0.7, 80)
    # Distinct muted colors for each candidate — visually separable but dim
    cand_colors = ["#c4a0b0", "#a0b8c4", "#c4c0a0", "#b0a0c4", "#c4a8a0"]
    cand_labels = ["Cand. 1", "Cand. 2", "Cand. 3", "Cand. 4", "Cand. 5"]
    for i in range(5):
        ax1.hist(
            arr[:, i],
            bins=bins,
            alpha=0.30,
            color=cand_colors[i],
            edgecolor="none",
            label=cand_labels[i],
        )

    ax1.hist(
        selected_scores,
        bins=bins,
        alpha=0.85,
        color="#2eaa4f",
        edgecolor="white",
        linewidth=0.3,
        label="Selected (best)",
    )
    ax1.set_xlabel("CLAP Score", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Selected vs. Individual Candidates", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")

    # --- Panel B: Best-of-N curve (mean CLAP as N increases) ---
    ax2 = axes[1]
    means_by_n: list[float] = []
    for n in range(1, 6):
        best_of_n = [float(np.max(row[:n])) for row in arr]
        means_by_n.append(statistics.mean(best_of_n))

    ax2.plot(range(1, 6), means_by_n, "o-", color="#2eaa4f", linewidth=2.5, markersize=8, zorder=3)
    for i, (n, val) in enumerate(zip(range(1, 6), means_by_n)):
        improvement = ""
        if i > 0:
            pct = (val - means_by_n[i - 1]) / abs(means_by_n[i - 1]) * 100
            improvement = f" (+{pct:.1f}%)"
        ax2.annotate(
            f"{val:.4f}{improvement}",
            (n, val),
            textcoords="offset points",
            xytext=(10, -5 if i % 2 == 0 else 10),
            fontsize=8,
            ha="left",
        )

    ax2.set_xlabel("N (candidates considered)", fontsize=11)
    ax2.set_ylabel("Mean CLAP Score", fontsize=11)
    ax2.set_title("Best-of-N Selection Curve", fontsize=10)
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "fig_best_of_n_analysis.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Benchmark comparison — CLAP + success rate + latency
# ---------------------------------------------------------------------------


def plot_benchmark_comparison(rows: list[dict]) -> None:
    by_model = group_by_model(rows)

    models = [m for m in MODEL_ORDER if m in by_model]
    labels = [MODEL_LABELS[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    # Compute stats
    mean_claps: list[float] = []
    std_claps: list[float] = []
    success_rates: list[float] = []
    median_latencies: list[float] = []

    for m in models:
        all_rows_for_model = [r for r in rows if r["model"] == m]
        succeeded = by_model[m]
        claps = [r["clap_reward"] for r in succeeded]
        mean_claps.append(statistics.mean(claps))
        std_claps.append(statistics.stdev(claps) if len(claps) > 1 else 0.0)
        success_rates.append(len(succeeded) / len(all_rows_for_model) * 100 if all_rows_for_model else 0)
        config_times = [r["config_gen_s"] for r in succeeded if r.get("config_gen_s") is not None]
        median_latencies.append(statistics.median(config_times) if config_times else 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Benchmark Comparison — 200 Test Prompts, 6 Controllers", fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(models))
    bar_width = 0.6

    # Panel 1: Mean CLAP score
    ax1 = axes[0]
    ax1.bar(x, mean_claps, bar_width, color=colors, alpha=ALPHA, edgecolor="white", linewidth=0.5,
            yerr=std_claps, error_kw={"linewidth": 0.8, "capsize": 3, "capthick": 0.8, "alpha": 0.5})
    # Random baseline reference line
    random_idx = models.index("random") if "random" in models else None
    if random_idx is not None:
        ax1.axhline(mean_claps[random_idx], color="#999999", linestyle=":", linewidth=1, alpha=0.7,
                     label=f"Random baseline ({mean_claps[random_idx]:.3f})")
        ax1.legend(fontsize=7, loc="upper left")
    ax1.set_ylabel("Mean CLAP Score", fontsize=10)
    ax1.set_title("CLAP Alignment", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    for i, v in enumerate(mean_claps):
        ax1.text(i, v + std_claps[i] + 0.008, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Panel 2: Schema success rate
    ax2 = axes[1]
    ax2.bar(x, success_rates, bar_width, color=colors, alpha=ALPHA, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Success Rate (%)", fontsize=10)
    ax2.set_title("Schema Validity", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax2.set_ylim(80, 102)
    ax2.axhline(100, color="#2eaa4f", linestyle=":", linewidth=0.8, alpha=0.5)
    for i, v in enumerate(success_rates):
        ax2.text(i, v + 0.3, f"{v:.0f}%", ha="center", fontsize=7, fontweight="bold")

    # Panel 3: Config generation latency (log scale)
    ax3 = axes[2]
    # Replace 0 latency for random with a small value for log scale
    plot_latencies = [max(lat, 0.001) for lat in median_latencies]
    ax3.bar(x, plot_latencies, bar_width, color=colors, alpha=ALPHA, edgecolor="white", linewidth=0.5)
    ax3.set_ylabel("Median Latency (s)", fontsize=10)
    ax3.set_title("Config Generation Latency", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax3.set_yscale("log")
    for i, v in enumerate(median_latencies):
        label = f"{v:.1f}s" if v >= 1 else (f"{v:.2f}s" if v > 0 else "0s")
        ax3.text(i, plot_latencies[i] * 1.3, label, ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig_benchmark_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading benchmark data...")
    bench_rows = load_benchmark_rows()
    print(f"  {len(bench_rows)} benchmark rows loaded")

    print("Loading HuggingFace scored dataset...")
    hf_rows = load_hf_scored_rows()
    print(f"  {len(hf_rows)} HF rows loaded")

    print("\nGenerating figures...")
    print("\n[1/5] Config generation latency histograms")
    plot_config_latency(bench_rows)

    print("\n[2/5] Audio synthesis latency histograms")
    plot_audio_latency(bench_rows)

    print("\n[3/5] HF dataset CLAP score distribution (selected best-of-5)")
    plot_hf_clap_distribution(hf_rows)

    print("\n[4/5] Best-of-N candidate analysis")
    plot_best_of_n_candidates(hf_rows)

    print("\n[5/5] Benchmark comparison (CLAP + success + latency)")
    plot_benchmark_comparison(bench_rows)

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
