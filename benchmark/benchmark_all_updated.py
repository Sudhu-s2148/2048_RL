import sys
import asyncio
import re
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

import game
import torch
import torch.nn as nn
import math
import csv
from helpers import valid_actions


TOTAL_EPISODES = 1000
EPSILON = 0

# Optional: specify only the networks you want to benchmark.
# Leave as None to benchmark every .pth checkpoint.
# Examples: ["network_4", "network_4.2"] or ["network_4.pth", "network_8.pth"]
BENCHMARK_NETWORKS = ["network_10"]


class BenchmarkNetwork(nn.Module):
    """Build a feed-forward network directly from a saved state_dict."""

    def __init__(self, state_dict):
        super().__init__()

        weights = [
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(".weight") and value.ndim == 2
        ]

        if not weights:
            raise ValueError("No Linear-layer weights found in checkpoint.")

        def natural_key(item):
            return [
                int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", item[0])
            ]

        weights.sort(key=natural_key)

        layers = []

        for index, (weight_key, weight) in enumerate(weights):
            bias_key = weight_key[:-6] + "bias"

            if bias_key not in state_dict:
                raise ValueError(
                    f"Missing bias for {weight_key} in checkpoint."
                )

            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(state_dict[bias_key])

            layers.append(linear)

            if index < len(weights) - 1:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

    def choice(self, state, masked=True):
        # state: flat list of 16 log2 values (from state_gen)
        batched = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        batched = batched.to(next(self.parameters()).device)

        with torch.no_grad():
            q = self.forward(batched).squeeze(0).tolist()   # flat list of 4 Q-values

        if masked:
            # masking uses the raw flat state, not the batched tensor
            q = valid_actions(torch.tensor(state, dtype=torch.float32), q)

        return int(max(range(4), key=lambda i: q[i]))


def load_network(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    network = BenchmarkNetwork(state_dict).to(device)
    network.eval()

    return network


def state_gen(board_state):
    transposed = [list(row) for row in zip(*board_state)]

    reduced_board = [
        [math.log2(val) if val != 0 else 0 for val in row]
        for row in transposed
    ]

    return torch.flatten(
        torch.tensor(reduced_board, dtype=torch.float32)
    ).tolist()


def max_tile(matrix):
    maximum = matrix[0][0]

    for row in matrix:
        for value in row:
            if value > maximum:
                maximum = value

    return maximum


def play_episode(network):
    board = game.Board()
    board.spawn_number()

    steps = 0

    while board.game_state:
        current_state = state_gen(board.board_state)
        action = network.choice(current_state)

        if action == 0:
            board.move_up()
        elif action == 1:
            board.move_left()
        elif action == 2:
            board.move_right()
        elif action == 3:
            board.move_down()
        else:
            raise ValueError(f"Invalid network action: {action}")

        next_state = state_gen(board.board_state)
        steps += 1

        # An unchanged board means the chosen action was invalid.
        if current_state == next_state:
            return {
                "score": board.score,
                "max_tile": max_tile(board.board_state),
                "steps": steps,
                "termination": "invalid_move"
            }

        board.is_game_over()

        if not board.game_state:
            return {
                "score": board.score,
                "max_tile": max_tile(board.board_state),
                "steps": steps,
                "termination": "game_over"
            }

    return {
        "score": board.score,
        "max_tile": max_tile(board.board_state),
        "steps": steps,
        "termination": "game_over"
    }


async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = project_dir / "artifacts" / "checkpoints"
    output_dir = project_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(checkpoint_dir.glob("*.pth"))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .pth checkpoints found in: {checkpoint_dir}"
        )

    if BENCHMARK_NETWORKS:
        requested = set(BENCHMARK_NETWORKS)
        requested_with_ext = {n if n.lower().endswith(".pth") else f"{n}.pth" for n in requested}
        checkpoint_paths = [p for p in checkpoint_paths if p.name in requested or p.name in requested_with_ext]
        if not checkpoint_paths:
            raise FileNotFoundError(f"None of the requested networks were found in: {checkpoint_dir}")
        print("Selected checkpoints:")
    else:
        print("No specific networks selected — benchmarking all checkpoints:")

    for path in checkpoint_paths:
        print(f"  - {path.name}")

    

    for checkpoint_path in checkpoint_paths:
        benchmark_rows = []
        print("\n" + "=" * 70)
        print(f"Benchmarking: {checkpoint_path.name}")
        print("=" * 70)

        network = load_network(checkpoint_path, device)

        invalid_terminations = 0
        game_over_terminations = 0
        scores = []

        for episode in range(TOTAL_EPISODES):
            result = play_episode(network)
            scores.append(result["score"])

            if result["termination"] == "invalid_move":
                invalid_terminations += 1
            else:
                game_over_terminations += 1

            print(
                f"Episode {episode + 1:4d} | "
                f"Score: {result['score']:6d} | "
                f"Max tile: {result['max_tile']:4d} | "
                f"Steps: {result['steps']:4d} | "
                f"Ended: {result['termination']}"
            )

            benchmark_rows.append([
                checkpoint_path.name,
                episode,
                result["score"],
                result["max_tile"],
                result["steps"],
                result["termination"]
            ])

        print(f"\n{checkpoint_path.name} summary:")
        print(f"  Average score:             {sum(scores) / len(scores):.2f}")
        print(f"  Best score:                {max(scores)}")
        print(f"  Invalid terminations:      {invalid_terminations}")
        print(f"  Game-over terminations:    {game_over_terminations}")

        save_path_csv = output_dir / f"benchmark_{checkpoint_path.name}.csv"

        with open(save_path_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "network",
                "episode",
                "score",
                "max_tile",
                "steps",
                "termination"
            ])
            writer.writerows(benchmark_rows)

    print("\n" + "=" * 70)
    print(f"Benchmark complete.")
    print(f"Results saved to: {save_path_csv}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
