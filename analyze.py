import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED = [
    "episode",
    "total_score",
    "max_tile",
    "ep_length",
    "valid_moves",
    "invalid_moves",
    "merging_moves",
    "non_merging_valid_moves",
    "random_invalid_moves",
    "random_valid_moves",
    "exploit_invalid_moves",
    "exploit_valid_moves",
    "epsilon"
]


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), None

    if path.suffix.lower() == ".json":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        columns = data.get("format", REQUIRED)

        return (
            pd.DataFrame(
                data.get("episodes", []),
                columns=columns
            ),
            data.get("parameters")
        )

    raise ValueError("Input must be a .csv or .json file.")


def summary(df, params):
    best_score = df["total_score"].idxmax()
    best_tile = df["max_tile"].idxmax()

    print("\n" + "=" * 60)
    print("2048 RL RUN ANALYSIS")
    print("=" * 60)

    print(f"Episodes:                {len(df):,}")
    print(f"Average score:           {df.total_score.mean():,.2f}")
    print(f"Median score:            {df.total_score.median():,.2f}")

    print(f"Best score:              {int(df.loc[best_score, 'total_score']):,}")
    print(f"  Episode:               {int(df.loc[best_score, 'episode'])}")

    print(f"Maximum tile:            {int(df.max_tile.max())}")
    print(f"  First reached:         {int(df.loc[best_tile, 'episode'])}")

    print(f"Average episode len:     {df.ep_length.mean():,.2f}")
    print(f"Median episode len:      {df.ep_length.median():,.2f}")

    print(f"Average valid moves:     {df.valid_moves.mean():,.2f}")
    print(f"Average invalid moves:   {df.invalid_moves.mean():,.2f}")
    print(f"Average merging moves:   {df.merging_moves.mean():,.2f}")
    print(f"Average non-merging v.:  {df.non_merging_valid_moves.mean():,.2f}")

    # ---------------------------------------------------------
    # Random vs exploitation
    # ---------------------------------------------------------

    total_random = (
        df.random_invalid_moves.sum()
        + df.random_valid_moves.sum()
    )

    total_exploit = (
        df.exploit_invalid_moves.sum()
        + df.exploit_valid_moves.sum()
    )

    if total_random > 0:
        random_invalid_rate = (
            df.random_invalid_moves.sum() / total_random * 100
        )
    else:
        random_invalid_rate = 0

    if total_exploit > 0:
        exploit_invalid_rate = (
            df.exploit_invalid_moves.sum() / total_exploit * 100
        )
    else:
        exploit_invalid_rate = 0

    print("\nRandom vs exploitation:")
    print(f"  Random moves:          {total_random:,}")
    print(f"    Invalid:             {int(df.random_invalid_moves.sum()):,}")
    print(f"    Valid:               {int(df.random_valid_moves.sum()):,}")
    print(f"    Invalid rate:        {random_invalid_rate:.2f}%")

    print(f"  Exploitation moves:    {total_exploit:,}")
    print(f"    Invalid:             {int(df.exploit_invalid_moves.sum()):,}")
    print(f"    Valid:               {int(df.exploit_valid_moves.sum()):,}")
    print(f"    Invalid rate:        {exploit_invalid_rate:.2f}%")

    print(f"\nFinal epsilon:           {df.epsilon.iloc[-1]:.6f}")

    # ---------------------------------------------------------
    # Tile distribution
    # ---------------------------------------------------------

    print("\nTile distribution:")

    for tile, count in df.max_tile.value_counts().sort_index().items():
        print(
            f"  {int(tile):>5}: "
            f"{count:>5} "
            f"({count / len(df) * 100:5.1f}%)"
        )

    if params:
        print("\nTraining parameters:")

        for key, value in params.items():
            print(f"  {key}: {value}")

    print("=" * 60)


def chunk_analysis(df, chunks=10):
    print("\nPerformance by training period:")

    size = max(1, len(df) // chunks)

    for start in range(0, len(df), size):

        part = df.iloc[start:start + size]

        print(
            f"  {int(part.episode.iloc[0]):>5}-"
            f"{int(part.episode.iloc[-1]):<5} | "
            f"avg score {part.total_score.mean():>8.1f} | "
            f"best {int(part.total_score.max()):>5} | "
            f"avg moves {part.ep_length.mean():>6.1f} | "
            f"best tile {int(part.max_tile.max()):>4}"
        )


def make_plots(df, output_dir, window):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.total_score,
        alpha=0.35,
        label="Episode score"
    )

    plt.plot(
        df.episode,
        df.total_score.rolling(window, min_periods=1).mean(),
        linewidth=2,
        label=f"{window}-episode moving average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("2048 RL — Score Progression")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "score_progression.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Best score
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.total_score.cummax(),
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Best score so far")
    plt.title("2048 RL — Best Score")
    plt.tight_layout()

    plt.savefig(
        output_dir / "best_score.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Max tile
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.max_tile,
        alpha=0.55
    )

    plt.xlabel("Episode")
    plt.ylabel("Maximum tile")
    plt.title("2048 RL — Maximum Tile Reached")
    plt.tight_layout()

    plt.savefig(
        output_dir / "max_tile.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Episode length
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.ep_length,
        alpha=0.4
    )

    plt.plot(
        df.episode,
        df.ep_length.rolling(window, min_periods=1).mean(),
        linewidth=2,
        label=f"{window}-episode moving average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Moves")
    plt.title("2048 RL — Episode Length")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "episode_length.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Move types
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.valid_moves.rolling(window, min_periods=1).mean(),
        label="Valid moves (avg)"
    )

    plt.plot(
        df.episode,
        df.invalid_moves.rolling(window, min_periods=1).mean(),
        label="Invalid moves (avg)"
    )

    plt.plot(
        df.episode,
        df.merging_moves.rolling(window, min_periods=1).mean(),
        label="Merging moves (avg)"
    )

    plt.plot(
        df.episode,
        df.non_merging_valid_moves.rolling(window, min_periods=1).mean(),
        label="Non-merging valid moves (avg)"
    )

    plt.xlabel("Episode")
    plt.ylabel("Move Count")
    plt.title(
        f"2048 RL — Move Types "
        f"({window}-episode moving average)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "move_types.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Random vs exploitation invalid moves
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.random_invalid_moves.rolling(
            window,
            min_periods=1
        ).mean(),
        label="Random invalid moves"
    )

    plt.plot(
        df.episode,
        df.exploit_invalid_moves.rolling(
            window,
            min_periods=1
        ).mean(),
        label="Exploitation invalid moves"
    )

    plt.xlabel("Episode")
    plt.ylabel("Invalid moves")
    plt.title(
        f"2048 RL — Invalid Moves by Action Source "
        f"({window}-episode moving average)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "invalid_moves_source.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Invalid move rate
    # ---------------------------------------------------------

    random_total = (
        df.random_invalid_moves
        + df.random_valid_moves
    )

    exploit_total = (
        df.exploit_invalid_moves
        + df.exploit_valid_moves
    )

    random_rate = (
        df.random_invalid_moves
        / random_total.replace(0, float("nan"))
    ) * 100

    exploit_rate = (
        df.exploit_invalid_moves
        / exploit_total.replace(0, float("nan"))
    ) * 100

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        random_rate.rolling(window, min_periods=1).mean(),
        label="Random invalid rate"
    )

    plt.plot(
        df.episode,
        exploit_rate.rolling(window, min_periods=1).mean(),
        label="Exploitation invalid rate"
    )

    plt.xlabel("Episode")
    plt.ylabel("Invalid move rate (%)")

    plt.title(
        f"2048 RL — Invalid Move Rate "
        f"({window}-episode moving average)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "invalid_move_rate.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Epsilon
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.episode,
        df.epsilon,
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("2048 RL — Exploration Decay")
    plt.tight_layout()

    plt.savefig(
        output_dir / "epsilon.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Tile distribution
    # ---------------------------------------------------------

    counts = df.max_tile.value_counts().sort_index()

    plt.figure(figsize=(10, 6))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.xlabel("Maximum tile reached")
    plt.ylabel("Number of episodes")
    plt.title("2048 RL — Maximum Tile Distribution")
    plt.tight_layout()

    plt.savefig(
        output_dir / "tile_distribution.png",
        dpi=150
    )

    plt.close()

    print(f"\nPlots saved to: {output_dir.resolve()}")


def main():

    parser = argparse.ArgumentParser(
        description="Analyze a 2048 RL training log."
    )

    parser.add_argument(
        "file",
        help="Path to CSV or JSON log"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window in episodes (default: 100)"
    )

    parser.add_argument(
        "--output",
        default="analysis",
        help="Directory for plots (default: analysis)"
    )

    args = parser.parse_args()

    df, params = load_data(args.file)

    missing = [
        c for c in REQUIRED
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "The log contains no episodes."
        )

    df = (
        df.sort_values("episode")
        .reset_index(drop=True)
    )

    summary(df, params)
    chunk_analysis(df)
    make_plots(
        df,
        args.output,
        args.window
    )


if __name__ == "__main__":
    main()