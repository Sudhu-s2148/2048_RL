import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_CSV_NAME = "network_comparison_mtl.csv"


def load_summary(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a .csv file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Benchmark summary CSV contains no rows: {path}")

    required = {
        "network", "architecture", "mask_mode", "episodes",
        "average_score", "median_score", "best_score",
        "average_max_tile", "median_max_tile", "average_steps",
        "median_steps", "invalid_terminations", "invalid_termination_pct",
        "game_over_terminations", "game_over_termination_pct",
        "total_mask_fallbacks", "validity_accuracy", "validity_precision",
        "validity_recall", "false_invalid_rate", "false_valid_rate",
        "validity_TP", "validity_FP", "validity_TN", "validity_FN",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} does not look like benchmark_summary.csv from "
            f"benchmark_mtl_updated.py.\nMissing columns: {sorted(missing)}"
        )
    return df


def collect_summary_paths(inputs, output_dir):
    """Accept summary CSV files or directories containing benchmark_summary.csv."""
    paths = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            summary_path = item / "benchmark_summary.csv"
            if summary_path.exists():
                paths.append(summary_path)
            else:
                # Also support a directory that contains multiple benchmark output folders.
                found = sorted(item.glob("**/benchmark_summary.csv"))
                if not found:
                    print(f"Warning: no benchmark_summary.csv found in {item}")
                paths.extend(found)
        else:
            paths.append(item)

    skip = (Path(output_dir).resolve() / OUTPUT_CSV_NAME)
    unique, seen = [], set()
    for path in paths:
        resolved = path.resolve()
        if resolved == skip or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    if not unique:
        raise FileNotFoundError(
            "No benchmark_summary.csv found. Run benchmark_mtl_updated.py first, "
            "then pass its output directory or the summary CSV."
        )

    frames = []
    for path in unique:
        frame = load_summary(path)
        frame["_source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_and_filter(df, networks=None, modes=None):
    df = df.copy()
    df["network"] = df["network"].astype(str)
    df["mask_mode"] = df["mask_mode"].astype(str)

    if networks:
        requested = {strip_suffix(n) for n in networks}
        names = df["network"].map(strip_suffix)
        df = df[names.isin(requested)]

    if modes:
        df = df[df["mask_mode"].isin(modes)]

    if df.empty:
        raise ValueError("No benchmark rows remain after applying the filters.")

    # Avoid ambiguous duplicate rows if multiple summary files were supplied.
    df = df.drop_duplicates(
        subset=["network", "architecture", "mask_mode", "episodes"],
        keep="last",
    )
    df["label"] = df["network"].map(strip_suffix) + " | " + df["mask_mode"]
    return df.reset_index(drop=True)


def strip_suffix(name):
    name = str(name).strip()
    for suffix in (".csv", ".pth"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    return name


def pct(value):
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"


def number(value, digits=1):
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def print_comparison(df):
    print("\\n" + "=" * 150)
    print("2048 RL — MTL BENCHMARK COMPARISON")
    print("=" * 150)

    display = pd.DataFrame({
        "network": df["network"],
        "architecture": df["architecture"],
        "mask_mode": df["mask_mode"],
        "episodes": df["episodes"],
        "avg score": df["average_score"].map(lambda x: number(x)),
        "median score": df["median_score"].map(lambda x: number(x)),
        "best score": df["best_score"].map(lambda x: number(x, 0)),
        "avg max tile": df["average_max_tile"].map(lambda x: number(x)),
        "avg steps": df["average_steps"].map(lambda x: number(x)),
        "invalid ends": df["invalid_terminations"],
        "invalid end rate": df["invalid_termination_pct"].map(pct),
        "game-over rate": df["game_over_termination_pct"].map(pct),
        "mask fallbacks": df["total_mask_fallbacks"],
    })
    print(display.to_string(index=False))
    print("=" * 150)

    mtl = df[df["architecture"].astype(str).str.startswith("MTL")]
    if not mtl.empty:
        validity = pd.DataFrame({
            "network": mtl["network"],
            "mask_mode": mtl["mask_mode"],
            "accuracy": mtl["validity_accuracy"].map(
                lambda x: pct(x * 100) if pd.notna(x) else "—"
            ),
            "precision": mtl["validity_precision"].map(
                lambda x: pct(x * 100) if pd.notna(x) else "—"
            ),
            "recall": mtl["validity_recall"].map(
                lambda x: pct(x * 100) if pd.notna(x) else "—"
            ),
            "false-invalid rate": mtl["false_invalid_rate"].map(
                lambda x: pct(x * 100) if pd.notna(x) else "—"
            ),
            "false-valid rate": mtl["false_valid_rate"].map(
                lambda x: pct(x * 100) if pd.notna(x) else "—"
            ),
            "TP": mtl["validity_TP"],
            "FP": mtl["validity_FP"],
            "TN": mtl["validity_TN"],
            "FN": mtl["validity_FN"],
        })
        print("\\nValidity-head diagnostics (same model predictions are evaluated in each mask mode):")
        print(validity.to_string(index=False))


def save_bar_plot(df, metric, ylabel, title, filename, output_dir, percent=False):
    values = pd.to_numeric(df[metric], errors="coerce")
    plot_df = df.loc[values.notna()].copy()
    if plot_df.empty:
        print(f"Skipping {filename}: no numeric values for {metric}.")
        return

    plt.figure(figsize=(max(10, len(plot_df) * 0.75), 6))
    plt.bar(plot_df["label"], pd.to_numeric(plot_df[metric], errors="coerce"))
    plt.xlabel("Network | mask mode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    if percent:
        plt.ylim(0, max(100, float(plot_df[metric].max()) * 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def make_plots(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_bar_plot(
        df, "average_score", "Average score", "2048 RL — Average Score",
        "average_score_comparison.png", output_dir
    )
    save_bar_plot(
        df, "median_score", "Median score", "2048 RL — Median Score",
        "median_score_comparison.png", output_dir
    )
    save_bar_plot(
        df, "best_score", "Best episode score", "2048 RL — Best Score",
        "best_score_comparison.png", output_dir
    )
    save_bar_plot(
        df, "average_max_tile", "Average maximum tile value",
        "2048 RL — Average Maximum Tile", "average_max_tile_comparison.png", output_dir
    )
    save_bar_plot(
        df, "invalid_termination_pct", "Episodes ending due to invalid move (%)",
        "2048 RL — Invalid-Move Termination Rate",
        "invalid_termination_comparison.png", output_dir, percent=True
    )
    save_bar_plot(
        df, "game_over_termination_pct", "Episodes ending at game over (%)",
        "2048 RL — Game-Over Termination Rate",
        "game_over_termination_comparison.png", output_dir, percent=True
    )
    save_bar_plot(
        df, "total_mask_fallbacks", "Fallback count",
        "2048 RL — Learned-Mask Fallbacks",
        "mask_fallback_comparison.png", output_dir
    )

    # Validity metrics are fractions (0–1) in benchmark_summary.csv, unlike the
    # termination *_pct columns, which are already percentages (0–100).
    mtl = df[df["architecture"].astype(str).str.startswith("MTL")].copy()
    if not mtl.empty:
        for metric, title, filename in [
            ("validity_accuracy", "Validity Head — Accuracy", "validity_accuracy.png"),
            ("validity_precision", "Validity Head — Precision", "validity_precision.png"),
            ("validity_recall", "Validity Head — Recall", "validity_recall.png"),
            ("false_invalid_rate", "Validity Head — False-Invalid Rate", "false_invalid_rate.png"),
            ("false_valid_rate", "Validity Head — False-Valid Rate", "false_valid_rate.png"),
        ]:
            mtl[metric] = pd.to_numeric(mtl[metric], errors="coerce") * 100
            save_bar_plot(
                mtl, metric, "Rate (%)", title, filename, output_dir, percent=True
            )

    print(f"\\nPlots and comparison CSV saved to: {output_dir.resolve()}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare networks and mask modes using benchmark_summary.csv "
            "created by benchmark_mtl_updated.py."
        )
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="benchmark_summary.csv file(s) and/or output directories. "
             "If omitted, searches for benchmark_summary.csv under this script's directory."
    )
    parser.add_argument(
        "--output", default=str(script_dir / "comparison_mtl"),
        help="Directory for comparison CSV and plots."
    )
    parser.add_argument(
        "--networks", nargs="*", default=None,
        help="Optional network names to include; .pth suffix is optional."
    )
    parser.add_argument(
        "--modes", nargs="*", choices=["none", "environment", "learned"],
        default=None,
        help="Optional mask modes to include."
    )
    args = parser.parse_args()

    inputs = args.inputs if args.inputs else [script_dir]
    all_data = collect_summary_paths(inputs, args.output)
    data = clean_and_filter(all_data, networks=args.networks, modes=args.modes)

    print("\\nNetworks and modes included:")
    for _, row in data.iterrows():
        print(f"  {row['network']} | {row['architecture']} | {row['mask_mode']} "
              f"| {row['episodes']} episodes")

    print_comparison(data)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [c for c in data.columns if not c.startswith("_")]
    data[export_columns].to_csv(output_dir / OUTPUT_CSV_NAME, index=False)
    make_plots(data, output_dir)


if __name__ == "__main__":
    main()
