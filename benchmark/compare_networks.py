import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_CSV_NAME = "network_comparison.csv"


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a .csv file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Benchmark CSV contains no episodes: {path}")
    return df


def find_column(df, candidates, description, source=""):
    for column in candidates:
        if column in df.columns:
            return column
    where = f" in {source}" if source else ""
    raise ValueError(
        f"Could not find the {description} column{where}.\n"
        f"Available columns: {list(df.columns)}"
    )


def collect_csv_paths(inputs, output_dir):
    """Expand the given files/directories into a de-duplicated list of CSV paths."""
    paths = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            found = sorted(item.glob("*.csv"))
            if not found:
                print(f"Warning: no .csv files found in {item}")
            paths.extend(found)
        else:
            paths.append(item)

    # Don't accidentally re-read our own summary output
    skip = (Path(output_dir).resolve() / OUTPUT_CSV_NAME)
    unique, seen = [], set()
    for p in paths:
        resolved = p.resolve()
        if resolved == skip or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(p)
    return unique


def summarize_one(df, name, source=""):
    score_col = find_column(df, ["total_score", "score"], "score", source)
    tile_col = find_column(df, ["max_tile", "best_tile"], "maximum tile", source)
    steps_col = find_column(df, ["steps", "ep_length", "episode_length"], "episode length", source)
    termination_col = find_column(
        df, ["termination_reason", "terminated_by", "termination"], "termination reason", source
    )

    reason = df[termination_col].astype(str).str.lower()
    invalid = reason.eq("invalid_move").sum()
    game_over = reason.eq("game_over").sum()
    total = len(df)

    return {
        "model": name,
        "episodes": total,
        "average_score": df[score_col].mean(),
        "median_score": df[score_col].median(),
        "best_score": df[score_col].max(),
        "average_steps": df[steps_col].mean(),
        "median_steps": df[steps_col].median(),
        "best_tile": df[tile_col].max(),
        "invalid_terminations": invalid,
        "invalid_rate": invalid / total * 100,
        "game_over_terminations": game_over,
        "game_over_rate": game_over / total * 100,
    }


def print_comparison(summary):
    print("\n" + "=" * 120)
    print("2048 RL — NETWORK BENCHMARK COMPARISON")
    print("=" * 120)

    display = summary.copy()
    display["average_score"] = display["average_score"].map(lambda x: f"{x:,.1f}")
    display["median_score"] = display["median_score"].map(lambda x: f"{x:,.1f}")
    display["best_score"] = display["best_score"].map(lambda x: f"{x:,.0f}")
    display["average_steps"] = display["average_steps"].map(lambda x: f"{x:,.1f}")
    display["median_steps"] = display["median_steps"].map(lambda x: f"{x:,.1f}")
    display["invalid_rate"] = display["invalid_rate"].map(lambda x: f"{x:.1f}%")
    display["game_over_rate"] = display["game_over_rate"].map(lambda x: f"{x:.1f}%")

    display = display[[
        "model", "episodes", "average_score", "median_score", "best_score",
        "average_steps", "median_steps", "best_tile", "invalid_terminations",
        "invalid_rate", "game_over_terminations", "game_over_rate"
    ]]
    print(display.to_string(index=False))
    print("=" * 120)


def make_plots(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["model"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, summary["average_score"])
    plt.xlabel("Network")
    plt.ylabel("Average score")
    plt.title("2048 RL — Average Score by Network")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "average_score_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, summary["best_score"])
    plt.xlabel("Network")
    plt.ylabel("Best score")
    plt.title("2048 RL — Best Score by Network")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "best_score_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, summary["best_tile"])
    plt.xlabel("Network")
    plt.ylabel("Maximum tile")
    plt.title("2048 RL — Maximum Tile by Network")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "best_tile_comparison.png", dpi=150)
    plt.close()

    x = list(range(len(summary)))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], summary["invalid_rate"], width=width, label="Invalid move")
    plt.bar([i + width / 2 for i in x], summary["game_over_rate"], width=width, label="Game over")
    plt.xlabel("Network")
    plt.ylabel("Termination rate (%)")
    plt.title("2048 RL — Episode Termination Reasons")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "termination_comparison.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir.resolve()}")


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compare 2048 RL networks, using one benchmark CSV per network."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Benchmark CSV files (one per network) and/or directories containing them. "
             "If omitted, all .csv files in the script's directory are used."
    )
    parser.add_argument(
        "--output",
        default=script_dir / "comparison",
        help="Directory for comparison plots and CSV"
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names, in the same order as the input files. "
             "By default each network is named after its CSV filename."
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        default=None,
        help="Only compare these networks (matched by name, .csv/.pth suffix optional). "
             "If omitted, all networks are used."
    )
    args = parser.parse_args()

    inputs = args.files if args.files else [script_dir]
    paths = collect_csv_paths(inputs, args.output)
    if not paths:
        raise FileNotFoundError("No benchmark CSV files found.")

    if args.names is not None:
        if len(args.names) != len(paths):
            raise ValueError(
                f"--names has {len(args.names)} entries but {len(paths)} CSV files were found."
            )
        names = list(args.names)
    else:
        names = [p.stem for p in paths]

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate network names detected. Rename the files or pass unique --names.\n"
            f"Names: {names}"
        )

    entries = list(zip(names, paths))

    if args.networks:
        requested = set()
        for n in args.networks:
            n = n.strip()
            for suffix in (".csv", ".pth"):
                if n.lower().endswith(suffix):
                    n = n[: -len(suffix)]
            requested.add(n)
        entries = [(n, p) for n, p in entries if n in requested]
        if not entries:
            raise ValueError(
                f"None of the requested networks were found: {', '.join(sorted(requested))}"
            )
        print("\nSelected networks:")
    else:
        print("\nNo networks specified — comparing all networks.")
        print("\nNetworks found:")
    for name, path in entries:
        print(f"  {name}  ({path})")

    rows = []
    for name, path in entries:
        df = load_data(path)
        rows.append(summarize_one(df, name, source=str(path)))
    summary_df = pd.DataFrame(rows)

    print_comparison(summary_df)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / OUTPUT_CSV_NAME, index=False)
    make_plots(summary_df, output_dir)


if __name__ == "__main__":
    main()
