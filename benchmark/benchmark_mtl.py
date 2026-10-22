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
import argparse
from helpers import valid_actions

parser = argparse.ArgumentParser()
parser.add_argument('--networks', nargs='+', type=str, default=['42'])
args = parser.parse_args()

TOTAL_EPISODES = 1000

# Optional: specify only the networks you want to benchmark.
# Leave as None to benchmark every .pth checkpoint.
BENCHMARK_NETWORKS = args.networks

# Action-selection mask mode:
# "none"        -> use raw Q-values, no validity mask
# "environment" -> use the environment's true validity mask
# "learned"     -> use the MTL network's learned validity head
MASK_MODE = "learned"
MODES = ["none", "environment", "learned"]

if MASK_MODE not in MODES:
    raise ValueError(
        "MASK_MODE must be 'none', 'environment', or 'learned'."
    )

accurate_mtl_mask = 0
total_masks = 0
class BenchmarkNetwork(nn.Module):
    """
    Loads either:
      1. the original vanilla DQN (single Q output), or
      2. the MTL DQN (shared layers + Q head + validity head).

    The architecture is detected from the checkpoint state_dict.
    """

    def __init__(self, state_dict):
        super().__init__()

        self.is_mtl = "V_output.weight" in state_dict

        if self.is_mtl:
            # Exact architecture used by the MTL Agent:
            # 16 -> 64 -> 64 -> 64 -> {Q: 4, V: 4}
            self.input = nn.Linear(
                state_dict["input.weight"].shape[1],
                state_dict["input.weight"].shape[0]
            )
            self.layer1 = nn.Linear(
                state_dict["layer1.weight"].shape[1],
                state_dict["layer1.weight"].shape[0]
            )
            self.layer2 = nn.Linear(
                state_dict["layer2.weight"].shape[1],
                state_dict["layer2.weight"].shape[0]
            )
            self.Q_output = nn.Linear(
                state_dict["Q_output.weight"].shape[1],
                state_dict["Q_output.weight"].shape[0]
            )
            self.V_output = nn.Linear(
                state_dict["V_output.weight"].shape[1],
                state_dict["V_output.weight"].shape[0]
            )

            for name in ["input", "layer1", "layer2", "Q_output", "V_output"]:
                layer = getattr(self, name)
                layer.weight.data.copy_(state_dict[f"{name}.weight"])
                layer.bias.data.copy_(state_dict[f"{name}.bias"])

        else:
            # Vanilla DQN checkpoint: reconstruct the sequential network
            # from its Linear-layer weights.
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
        if self.is_mtl:
            x = torch.relu(self.input(state))
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))

            q = self.Q_output(x)
            raw_v = self.V_output(x)

            return q, raw_v

        return self.network(state), None

    def choice(self, state, mask_mode="learned"):
        """
        Deterministic benchmark policy.

        mask_mode="none":
            Use raw Q-values with no validity mask.

        mask_mode="environment":
            Use the environment's true validity mask.

        mask_mode="learned":
            Use the MTL network's learned validity head:
            sigmoid(V) -> threshold at 0.5 -> mask invalid Q values.

        Returns:
            action, fallback, predicted_mask
        """
        if mask_mode not in {"none", "environment", "learned"}:
            raise ValueError(
                "mask_mode must be 'none', 'environment', or 'learned'."
            )

        batched = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        batched = batched.to(next(self.parameters()).device)

        with torch.no_grad():
            q, raw_v = self.forward(batched)
            q = q.squeeze(0)

            # Vanilla DQN has no validity head.
            if not self.is_mtl:
                action = torch.argmax(q).item()
                return int(action), False, None

            # No mask: directly select the highest Q-value.
            if mask_mode == "none":
                action = torch.argmax(q).item()
                return int(action), False, None

            if mask_mode == "environment":
                # Environment/oracle mask.
                validity_mask = torch.tensor(
                    valid_actions(state),
                    dtype=torch.bool,
                    device=q.device
                )
                env_V = validity_mask
                fallback = False

            else:  # mask_mode == "learned"
                # Learned validity mask from the MTL V-head.
                validity_prob = torch.sigmoid(raw_v.squeeze(0))
                validity_mask = validity_prob >= 0.5
                mtl_V = validity_mask

                # The learned head may predict every action as invalid.
                # Fall back to unmasked Q for this benchmark step and record it.
                fallback = not bool(validity_mask.any())

            if fallback:
                action = torch.argmax(q).item()
            else:
                masked_q = q.masked_fill(~validity_mask, float("-inf"))
                action = torch.argmax(masked_q).item()

            validity_mask = torch.tensor(
                valid_actions(state),
                dtype=torch.bool,
                device=q.device
            )
            env_V = validity_mask
            validity_prob = torch.sigmoid(raw_v.squeeze(0))
            mtl_V = validity_prob >= 0.5

            global accurate_mtl_mask, total_masks
            if torch.equal(mtl_V,env_V): accurate_mtl_mask+=1
            total_masks+=1

            # The learned head may predict every action as invalid.
            # Fall back to unmasked Q for this benchmark step and record it.
            fallback = not bool(validity_mask.any())

            #print("env:",env_V)
            #print("mtl:",mtl_V)

            return int(action), fallback, validity_mask.cpu().tolist()
        
def validity_percentage(total,accurate):
    return (accurate/total)*100


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
    validity_fallbacks = 0
    validity_correct = 0
    validity_total = 0

    while board.game_state:
        current_state = state_gen(board.board_state)

        action, fallback, predicted_mask = network.choice(
            current_state,
            mask_mode=MASK_MODE
        )

        if fallback:
            validity_fallbacks += 1

        # For an MTL model, compare the learned validity prediction
        # against the environment's true mask for evaluation only.
        if network.is_mtl and MASK_MODE == "learned":
            true_mask = valid_actions(current_state)

            validity_correct += sum(
                int(pred == true)
                for pred, true in zip(predicted_mask, true_mask)
            )
            validity_total += len(true_mask)

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
                "termination": "invalid_move",
                "validity_fallbacks": validity_fallbacks,
                "validity_correct": validity_correct,
                "validity_total": validity_total
            }

        board.is_game_over()

        if not board.game_state:
            return {
                "score": board.score,
                "max_tile": max_tile(board.board_state),
                "steps": steps,
                "termination": "game_over",
                "validity_fallbacks": validity_fallbacks,
                "validity_correct": validity_correct,
                "validity_total": validity_total
            }

    return {
        "score": board.score,
        "max_tile": max_tile(board.board_state),
        "steps": steps,
        "termination": "game_over",
        "validity_fallbacks": validity_fallbacks,
        "validity_correct": validity_correct,
        "validity_total": validity_total
    }


async def main():
    global MASK_MODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = project_dir / "artifacts" / "MTL_DQN" / "checkpoints"
    output_dir = project_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(checkpoint_dir.glob("*.pth"))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .pth checkpoints found in: {checkpoint_dir}"
        )

    if BENCHMARK_NETWORKS:
        requested = set(BENCHMARK_NETWORKS)
        requested_with_ext = {
            n if n.lower().endswith(".pth") else f"{n}.pth"
            for n in requested
        }

        checkpoint_paths = [
            p for p in checkpoint_paths
            if p.name in requested or p.name in requested_with_ext
        ]

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"None of the requested networks were found in: {checkpoint_dir}"
            )

        print("Selected checkpoints:")
    else:
        print("No specific networks selected — benchmarking all checkpoints:")

    for path in checkpoint_paths:
        print(f"  - {path.name}")

    last_save_path = None

    for checkpoint_path in checkpoint_paths:
        benchmark_rows = []

        

        network = load_network(checkpoint_path, device)

        model_type = "MTL DQN" if network.is_mtl else "Vanilla DQN"
        print(f"Model type: {model_type}")
        if network.is_mtl:
            print(f"Action mask: {MASK_MODE}")

        
        print("\n" + "=" * 70)
        print(f"Benchmarking: {checkpoint_path.name}")
        print("=" * 70)

        invalid_terminations = 0
        game_over_terminations = 0
        total_fallbacks = 0
        total_validity_correct = 0
        total_validity_predictions = 0
        scores = []

        for episode in range(TOTAL_EPISODES):
            result = play_episode(network)
            scores.append(result["score"])

            total_fallbacks += result["validity_fallbacks"]
            total_validity_correct += result["validity_correct"]
            total_validity_predictions += result["validity_total"]

            if result["termination"] == "invalid_move":
                invalid_terminations += 1
            else:
                game_over_terminations += 1

            print(
                f"Episode {episode + 1:4d} | "
                f"Score: {result['score']:6d} | "
                f"Max tile: {result['max_tile']:4d} | "
                f"Steps: {result['steps']:4d} | "
                f"Fallbacks: {result['validity_fallbacks']:3d} | "
                f"Ended: {result['termination']}"
            )

            benchmark_rows.append([
                checkpoint_path.name,
                episode,
                result["score"],
                result["max_tile"],
                result["steps"],
                result["termination"],
                result["validity_fallbacks"],
                result["validity_correct"],
                result["validity_total"]
            ])

        print(f"\n{checkpoint_path.name} summary:")
        print(f"  Model type:                {model_type}")
        print(f"  Average score:             {sum(scores) / len(scores):.2f}")
        print(f"  Best score:                {max(scores)}")
        print(f"  Invalid terminations:      {invalid_terminations}")
        print(f"  Game-over terminations:    {game_over_terminations}")

        if network.is_mtl and MASK_MODE == "learned":
            validity_accuracy = (
                total_validity_correct / total_validity_predictions
                if total_validity_predictions
                else 0.0
            )

            print(f"  Validity accuracy:         {validity_accuracy * 100:.2f}%")
            print(f"  Validity fallbacks:        {total_fallbacks}")
            print(f" my valdity accuracy:        {validity_percentage(total_masks,accurate_mtl_mask)}")

        save_path_csv = (
            output_dir / f"mtl_benchmark_{checkpoint_path.name}_mtlmask.csv"
        )

        with open(save_path_csv, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "network",
                "episode",
                "score",
                "max_tile",
                "steps",
                "termination",
                "validity_fallbacks",
                "validity_correct",
                "validity_total"
            ])

            writer.writerows(benchmark_rows)

        last_save_path = save_path_csv

        print("\n" + "=" * 70)
        print("Benchmark complete.")
        print(f"Results saved to: {last_save_path}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
