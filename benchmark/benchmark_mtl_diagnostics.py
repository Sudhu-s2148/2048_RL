import argparse
import csv
import math
import random
import re
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Project layout expected:
# project/
#   game.py
#   helpers.py
#   artifacts/MTL_DQN/checkpoints/*.pth
#   benchmark/  (outputs)
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import game
from helpers import valid_actions


DEFAULT_MODES = ["none", "environment", "learned"]
VALID_MODES = set(DEFAULT_MODES)


class BenchmarkNetwork(nn.Module):
    """Reconstruct a vanilla DQN or either version of the MTL network."""

    def __init__(self, state_dict):
        super().__init__()
        self.is_mtl = "V_output.weight" in state_dict
        self.has_extra_validity_layer = "V_layer.weight" in state_dict

        if self.is_mtl:
            self.input = self._make_linear(state_dict, "input")
            self.layer1 = self._make_linear(state_dict, "layer1")
            self.layer2 = self._make_linear(state_dict, "layer2")
            self.Q_output = self._make_linear(state_dict, "Q_output")
            self.V_output = self._make_linear(state_dict, "V_output")

            if self.has_extra_validity_layer:
                self.V_layer = self._make_linear(state_dict, "V_layer")

            for name in ["input", "layer1", "layer2", "Q_output", "V_output"]:
                self._copy_linear(getattr(self, name), state_dict, name)
            if self.has_extra_validity_layer:
                self._copy_linear(self.V_layer, state_dict, "V_layer")
        else:
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
                    raise ValueError(f"Missing bias for {weight_key} in checkpoint.")
                linear = nn.Linear(weight.shape[1], weight.shape[0])
                linear.weight.data.copy_(weight)
                linear.bias.data.copy_(state_dict[bias_key])
                layers.append(linear)
                if index < len(weights) - 1:
                    layers.append(nn.ReLU())
            self.network = nn.Sequential(*layers)

    @staticmethod
    def _make_linear(state_dict, name):
        weight = state_dict[f"{name}.weight"]
        return nn.Linear(weight.shape[1], weight.shape[0])

    @staticmethod
    def _copy_linear(layer, state_dict, name):
        layer.weight.data.copy_(state_dict[f"{name}.weight"])
        layer.bias.data.copy_(state_dict[f"{name}.bias"])

    def forward(self, state):
        if not self.is_mtl:
            return self.network(state), None

        x = torch.relu(self.input(state))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        q = self.Q_output(x)

        # Older checkpoints use the shared trunk directly.
        # Newer checkpoints have a dedicated extra layer in the validity branch.
        if self.has_extra_validity_layer:
            validity_features = torch.relu(self.V_layer(x))
            raw_v = self.V_output(validity_features)
        else:
            raw_v = self.V_output(x)

        return q, raw_v

    def predict_validity_mask(self, state):
        """Return the learned boolean mask, or None for vanilla DQN."""
        if not self.is_mtl:
            return None
        device = next(self.parameters()).device
        batch = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, raw_v = self.forward(batch)
            probabilities = torch.sigmoid(raw_v.squeeze(0))
            return (probabilities >= 0.5).cpu().tolist()

    def choice(self, state, mask_mode):
        if mask_mode not in VALID_MODES:
            raise ValueError(f"Unknown mask mode: {mask_mode}")

        device = next(self.parameters()).device
        batch = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q, raw_v = self.forward(batch)
            q = q.squeeze(0)

            # Vanilla DQN has no validity head, so it always uses raw Q-values.
            if not self.is_mtl or mask_mode == "none":
                return int(torch.argmax(q).item()), False

            if mask_mode == "environment":
                mask = torch.tensor(
                    valid_actions(state), dtype=torch.bool, device=q.device
                )
            else:
                mask = torch.sigmoid(raw_v.squeeze(0)) >= 0.5

            # If the learned head rejects every action, use raw Q-values as a
            # safe fallback and count this event.
            fallback = not bool(mask.any())
            if fallback:
                action = torch.argmax(q).item()
            else:
                masked_q = q.masked_fill(~mask, float("-inf"))
                action = torch.argmax(masked_q).item()

            return int(action), fallback


def state_gen(board_state):
    transposed = [list(row) for row in zip(*board_state)]
    reduced_board = [
        [math.log2(value) if value != 0 else 0 for value in row]
        for row in transposed
    ]
    return torch.flatten(torch.tensor(reduced_board, dtype=torch.float32)).tolist()


def max_tile(matrix):
    return max(value for row in matrix for value in row)


def load_network(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    network = BenchmarkNetwork(state_dict).to(device)
    network.eval()
    return network


def play_episode(network, mask_mode):
    board = game.Board()
    board.spawn_number()

    steps = 0
    fallbacks = 0
    validity_tp = validity_fp = validity_tn = validity_fn = 0

    while board.game_state:
        current_state = state_gen(board.board_state)

        # Evaluate the learned validity head against the environment on every
        # visited state, regardless of which action-selection mode is active.
        if network.is_mtl:
            predicted = network.predict_validity_mask(current_state)
            truth = [bool(value) for value in valid_actions(current_state)]
            for pred, actual in zip(predicted, truth):
                if pred and actual:
                    validity_tp += 1
                elif pred and not actual:
                    validity_fp += 1
                elif not pred and not actual:
                    validity_tn += 1
                else:
                    validity_fn += 1

        action, fallback = network.choice(current_state, mask_mode)
        fallbacks += int(fallback)

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

        steps += 1
        next_state = state_gen(board.board_state)

        if current_state == next_state:
            termination = "invalid_move"
            break

        board.is_game_over()
        if not board.game_state:
            termination = "game_over"
            break
    else:
        termination = "game_over"

    return {
        "score": board.score,
        "max_tile": max_tile(board.board_state),
        "steps": steps,
        "termination": termination,
        "validity_fallbacks": fallbacks,
        "validity_tp": validity_tp,
        "validity_fp": validity_fp,
        "validity_tn": validity_tn,
        "validity_fn": validity_fn,
    }


def summarize(checkpoint_name, model, mode, rows):
    scores = [row["score"] for row in rows]
    steps = [row["steps"] for row in rows]
    max_tiles = [row["max_tile"] for row in rows]
    invalid_ends = sum(row["termination"] == "invalid_move" for row in rows)
    game_over_ends = len(rows) - invalid_ends

    tp = sum(row["validity_tp"] for row in rows)
    fp = sum(row["validity_fp"] for row in rows)
    tn = sum(row["validity_tn"] for row in rows)
    fn = sum(row["validity_fn"] for row in rows)
    total_predictions = tp + fp + tn + fn

    accuracy = (tp + tn) / total_predictions if total_predictions else ""
    precision = tp / (tp + fp) if tp + fp else ""
    recall = tp / (tp + fn) if tp + fn else ""
    false_invalid_rate = fn / (tp + fn) if tp + fn else ""
    false_valid_rate = fp / (fp + tn) if fp + tn else ""

    return {
        "network": checkpoint_name,
        "architecture": (
            "MTL + extra V layer"
            if model.is_mtl and model.has_extra_validity_layer
            else "MTL original"
            if model.is_mtl
            else "Vanilla DQN"
        ),
        "mask_mode": mode,
        "episodes": len(rows),
        "average_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "best_score": max(scores),
        "average_max_tile": statistics.mean(max_tiles),
        "median_max_tile": statistics.median(max_tiles),
        "average_steps": statistics.mean(steps),
        "median_steps": statistics.median(steps),
        "invalid_terminations": invalid_ends,
        "invalid_termination_pct": 100 * invalid_ends / len(rows),
        "game_over_terminations": game_over_ends,
        "game_over_termination_pct": 100 * game_over_ends / len(rows),
        "total_mask_fallbacks": sum(row["validity_fallbacks"] for row in rows),
        "validity_accuracy": accuracy,
        "validity_precision": precision,
        "validity_recall": recall,
        "false_invalid_rate": false_invalid_rate,
        "false_valid_rate": false_valid_rate,
        "validity_TP": tp,
        "validity_FP": fp,
        "validity_TN": tn,
        "validity_FN": fn,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DQN/MTL checkpoints with multiple action-mask modes."
    )
    parser.add_argument(
        "--networks", nargs="*", default=None,
        help="Checkpoint names with or without .pth. Omit to benchmark all checkpoints."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES,
        help="Action-selection modes to benchmark."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional Python/PyTorch seed. Reset to this seed for each network/mode pair."
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = PROJECT_DIR / "artifacts" / "MTL_DQN" / "checkpoints"
    output_dir = PROJECT_DIR / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(checkpoint_dir.glob("*.pth"))
    if args.networks:
        requested = set(args.networks)
        requested_with_ext = {
            name if name.lower().endswith(".pth") else f"{name}.pth"
            for name in requested
        }
        checkpoint_paths = [
            path for path in checkpoint_paths
            if path.name in requested or path.name in requested_with_ext
        ]

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No matching .pth checkpoints found in {checkpoint_dir}"
        )

    print(f"Device: {device}")
    print(f"Episodes per network/mode: {args.episodes}")
    print(f"Modes: {', '.join(args.modes)}")
    print("Checkpoints:")
    for path in checkpoint_paths:
        print(f"  - {path.name}")

    summary_rows = []
    episode_fields = [
        "network", "architecture", "mask_mode", "episode", "score", "max_tile",
        "steps", "termination", "validity_fallbacks", "validity_tp",
        "validity_fp", "validity_tn", "validity_fn"
    ]

    for checkpoint_path in checkpoint_paths:
        model = load_network(checkpoint_path, device)
        architecture = (
            "MTL + extra V layer"
            if model.is_mtl and model.has_extra_validity_layer
            else "MTL original"
            if model.is_mtl
            else "Vanilla DQN"
        )

        modes = args.modes if model.is_mtl else ["none"]
        for mode in modes:
            if args.seed is not None:
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)

            print(f"\n{'=' * 72}")
            print(f"Checkpoint: {checkpoint_path.name} | Architecture: {architecture}")
            print(f"Mask mode: {mode} | Episodes: {args.episodes}")
            print(f"{'=' * 72}")

            episode_rows = []
            for episode in range(args.episodes):
                result = play_episode(model, mode)
                row = {
                    "network": checkpoint_path.name,
                    "architecture": architecture,
                    "mask_mode": mode,
                    "episode": episode + 1,
                    **result,
                }
                episode_rows.append(row)
                print(
                    f"Episode {episode + 1:4d} | "
                    f"Score: {result['score']:6d} | "
                    f"Max tile: {result['max_tile']:4d} | "
                    f"Steps: {result['steps']:4d} | "
                    f"Fallbacks: {result['validity_fallbacks']:3d} | "
                    f"Ended: {result['termination']}"
                )

            detail_path = output_dir / (
                f"mtl_benchmark_{checkpoint_path.stem}_{mode}.csv"
            )
            write_csv(detail_path, episode_rows, episode_fields)

            summary = summarize(checkpoint_path.name, model, mode, episode_rows)
            summary_rows.append(summary)

            print("\nSummary:")
            for key in [
                "average_score", "median_score", "best_score", "average_max_tile",
                "average_steps", "invalid_termination_pct",
                "game_over_termination_pct", "total_mask_fallbacks",
                "validity_accuracy", "validity_precision", "validity_recall",
                "false_invalid_rate", "false_valid_rate"
            ]:
                value = summary[key]
                if isinstance(value, float):
                    if key.endswith("_pct") or key.endswith("_rate") or key.startswith("validity_"):
                        print(f"  {key}: {value * 100:.2f}%")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            print(f"Per-episode CSV: {detail_path}")

    summary_fields = list(summary_rows[0].keys())
    summary_path = output_dir / "benchmark_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"\nAll benchmarks complete.")
    print(f"Combined summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
