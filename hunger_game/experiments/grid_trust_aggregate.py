import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import csv


def load_seed_csvs(base_dir: str) -> List[str]:
    """Return sorted list of per-seed CSV paths."""
    pattern = os.path.join(base_dir, "grid_stats_seed*.csv")
    paths = sorted(glob.glob(pattern))
    return paths


def load_metric_arrays(csv_paths: List[str], metric_cols: List[str]):
    """
    Load specified columns from each per-seed CSV and stack into
    arrays of shape [num_seeds, num_episodes].
    """
    if not csv_paths:
        raise RuntimeError("No per-seed CSV files found (grid_stats_seed*.csv).")

    # First, read header from the first file to get column indices
    with open(csv_paths[0], "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    col_idx = {name: header.index(name) for name in metric_cols}

    all_metrics = {name: [] for name in metric_cols}

    for path in csv_paths:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip header
            rows = list(reader)

        for name in metric_cols:
            idx = col_idx[name]
            vals = []
            for row in rows:
                if idx >= len(row) or row[idx] == "":
                    vals.append(np.nan)
                else:
                    vals.append(float(row[idx]))
            all_metrics[name].append(np.asarray(vals, dtype=np.float32))

    # Stack per-seed arrays -> [num_seeds, num_episodes]
    for name in metric_cols:
        all_metrics[name] = np.stack(all_metrics[name], axis=0)

    return all_metrics


def _smooth(arr: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Simple moving average smoothing over the time axis.

    This is applied to the mean curves over episodes to make long-term trends
    easier to see, without changing the cross-seed variability information
    captured by the error bands (SEM).
    """
    if window is None or window <= 1:
        return arr
    if arr.size == 0 or arr.size < window:
        return arr
    out = np.empty_like(arr)
    for i in range(arr.size):
        s = max(0, i - window + 1)
        out[i] = float(np.mean(arr[s : i + 1]))
    return out


def plot_with_shaded_error(
    x: np.ndarray,
    mean: np.ndarray,
    err: np.ndarray,
    ax: plt.Axes,
    color: str,
    label: str,
    alpha: float = 0.2,
):
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(
        x,
        mean - err,
        mean + err,
        color=color,
        alpha=alpha,
        linewidth=0.0,
    )


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    csv_paths = load_seed_csvs(base_dir)
    print(f"Found {len(csv_paths)} per-seed CSV files.")

    metric_cols = [
        "trust_cooldown",
        "base_cooldown",
        "trust_conflict",
        "base_conflict",
        "trust_min_return",
        "base_min_return",
        "trust_total_resources",
        "base_total_resources",
        "trust_gini_rewards",
        "base_gini_rewards",
    ]

    data = load_metric_arrays(csv_paths, metric_cols)
    num_seeds, num_eps = data["trust_cooldown"].shape
    episodes = np.arange(1, num_eps + 1)

    # Compute mean / std / sem over seeds, per episode
    stats = {}
    for name, arr in data.items():
        # arr: [num_seeds, num_episodes]
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        sem = std / np.sqrt(num_seeds)
        stats[name] = {"mean": mean, "std": std, "sem": sem}

    # 1) Average cooldown + conflict (trust vs baseline) with shaded SEM
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # cooldown
    ax = axes[0]
    # apply temporal smoothing to the mean curves for readability
    m_trust = _smooth(stats["trust_cooldown"]["mean"], window=100)
    se_trust = stats["trust_cooldown"]["sem"]
    m_base = _smooth(stats["base_cooldown"]["mean"], window=100)
    se_base = stats["base_cooldown"]["sem"]
    plot_with_shaded_error(
        episodes,
        m_trust,
        se_trust,
        ax,
        color="tab:red",
        label="Trust-based",
    )
    plot_with_shaded_error(
        episodes,
        m_base,
        se_base,
        ax,
        color="tab:orange",
        label="Baseline",
    )
    ax.set_title("Cooldown fraction (mean ± SEM over 30 seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cooldown fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 紧一点的 y 轴范围，让两条线的差异更明显
    ymin_cd = min(m_trust.min(), m_base.min()) - 0.002
    ymax_cd = max(m_trust.max(), m_base.max()) + 0.002
    ax.set_ylim(ymin_cd, ymax_cd)

    # conflict
    ax = axes[1]
    m_trust_c = _smooth(stats["trust_conflict"]["mean"], window=100)
    se_trust_c = stats["trust_conflict"]["sem"]
    m_base_c = _smooth(stats["base_conflict"]["mean"], window=100)
    se_base_c = stats["base_conflict"]["sem"]
    plot_with_shaded_error(
        episodes,
        m_trust_c,
        se_trust_c,
        ax,
        color="tab:green",
        label="Conflict (trust)",
    )
    plot_with_shaded_error(
        episodes,
        m_base_c,
        se_base_c,
        ax,
        color="tab:blue",
        label="Conflict (baseline)",
    )
    ax.set_title("Conflict level (mean ± SEM over 30 seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg conflicts per step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 冲突的范围更窄一点（类似 0.008–0.018），让差异更清楚
    ymin_conf = min(m_trust_c.min(), m_base_c.min()) - 0.002
    ymax_conf = max(m_trust_c.max(), m_base_c.max()) + 0.002
    ax.set_ylim(ymin_conf, ymax_conf)

    plt.tight_layout()
    out_path = os.path.join(base_dir, "grid_trust_results_mean.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated cooldown/conflict figure to: {out_path}")

    # 2) Scalar summaries over episodes: min return, Gini, total resources
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # min return
    ax = axes[0]
    m_trust_mr = _smooth(stats["trust_min_return"]["mean"], window=100)
    se_trust_mr = stats["trust_min_return"]["sem"]
    m_base_mr = _smooth(stats["base_min_return"]["mean"], window=100)
    se_base_mr = stats["base_min_return"]["sem"]
    plot_with_shaded_error(
        episodes,
        m_trust_mr,
        se_trust_mr,
        ax,
        color="tab:red",
        label="Trust-based",
    )
    plot_with_shaded_error(
        episodes,
        m_base_mr,
        se_base_mr,
        ax,
        color="tab:blue",
        label="Baseline",
    )
    ax.set_title("Minimum return per episode (mean ± SEM)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Min return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # total resources
    ax = axes[1]
    m_trust_res = _smooth(stats["trust_total_resources"]["mean"], window=100)
    se_trust_res = stats["trust_total_resources"]["sem"]
    m_base_res = _smooth(stats["base_total_resources"]["mean"], window=100)
    se_base_res = stats["base_total_resources"]["sem"]
    plot_with_shaded_error(
        episodes,
        m_trust_res,
        se_trust_res,
        ax,
        color="tab:red",
        label="Trust-based",
    )
    plot_with_shaded_error(
        episodes,
        m_base_res,
        se_base_res,
        ax,
        color="tab:blue",
        label="Baseline",
    )
    ax.set_title("Remaining resources at end of episode (mean ± SEM)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total resources")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Gini
    ax = axes[2]
    m_trust_g = _smooth(stats["trust_gini_rewards"]["mean"], window=100)
    se_trust_g = stats["trust_gini_rewards"]["sem"]
    m_base_g = _smooth(stats["base_gini_rewards"]["mean"], window=100)
    se_base_g = stats["base_gini_rewards"]["sem"]
    plot_with_shaded_error(
        episodes,
        m_trust_g,
        se_trust_g,
        ax,
        color="tab:red",
        label="Trust-based",
    )
    plot_with_shaded_error(
        episodes,
        m_base_g,
        se_base_g,
        ax,
        color="tab:blue",
        label="Baseline",
    )
    ax.set_title("Reward Gini coefficient (mean ± SEM)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gini")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(base_dir, "grid_trust_stats_mean.png")
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated scalar-metric figure to: {out_path2}")

    # 3) Difference curves: baseline - trust（额外一张“差值图”）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # cooldown difference
    ax = axes[0]
    diff_cd = _smooth(
        stats["base_cooldown"]["mean"] - stats["trust_cooldown"]["mean"], window=100
    )
    ax.plot(episodes, diff_cd, color="tab:purple", linewidth=1.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Cooldown difference (baseline - trust)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Δ cooldown")
    ax.grid(alpha=0.3)

    # conflict difference
    ax = axes[1]
    diff_conf = _smooth(
        stats["base_conflict"]["mean"] - stats["trust_conflict"]["mean"], window=100
    )
    ax.plot(episodes, diff_conf, color="tab:brown", linewidth=1.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Conflict difference (baseline - trust)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Δ conflict")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path3 = os.path.join(base_dir, "grid_trust_diff_mean.png")
    plt.savefig(out_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved difference figure to: {out_path3}")


if __name__ == "__main__":
    main()


