"""2048 vanilla DDQN diagnostics; compatible with compare_mtl_diagnostics.py.

Run from any directory. Default checkpoint location:
  <project>/artifacts/vanilla_DDQN/checkpoints
Use --checkpoint-path to point to a specific .pth file.
"""
import argparse
import csv
import math
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import game
from helpers import valid_actions


class DDQNNetwork(nn.Module):
    """Exact inference architecture of the supplied vanilla DDQN agent."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
        self.Q_output = nn.Linear(64, 4)
        self.architecture = 'Vanilla DDQN: 16-64-64-64-4'

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.Q_output(x)


def load_network(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    model = DDQNNetwork()
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


def state_gen(board_state):
    # Same transposed log2 representation as the supplied MTL benchmark.
    transposed = zip(*board_state)
    return [math.log2(value) if value else 0.0 for row in transposed for value in row]


def choose_action(model, state, mode):
    device = next(model.parameters()).device
    with torch.no_grad():
        q = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
    if mode == 'environment':
        mask = torch.tensor(valid_actions(state), dtype=torch.bool, device=device)
        if mask.any():
            q = q.masked_fill(~mask, float('-inf'))
    return int(q.argmax().item())


def play_episode(model, mode, max_steps):
    board = game.Board()
    board.spawn_number()
    steps = invalid_moves = 0
    termination = 'game_over'
    while board.game_state:
        if steps >= max_steps:
            termination = 'step_cap'
            break
        state = state_gen(board.board_state)
        action = choose_action(model, state, mode)
        if action == 0:
            board.move_up()
        elif action == 1:
            board.move_left()
        elif action == 2:
            board.move_right()
        else:
            board.move_down()
        steps += 1
        if state == state_gen(board.board_state):
            invalid_moves += 1
            termination = 'invalid_move'
            break
        board.is_game_over()
    return {
        'score': board.score,
        'max_tile': max(max(row) for row in board.board_state),
        'steps': steps,
        'termination': termination,
        'invalid_moves': invalid_moves,
        'validity_fallbacks': 0,
        'validity_tp': 0, 'validity_fp': 0, 'validity_tn': 0, 'validity_fn': 0,
    }


def summarize(name, mode, rows):
    scores = [r['score'] for r in rows]
    tiles = [r['max_tile'] for r in rows]
    steps = [r['steps'] for r in rows]
    n = len(rows)
    invalid = sum(r['termination'] == 'invalid_move' for r in rows)
    gameovers = sum(r['termination'] == 'game_over' for r in rows)
    capped = n - invalid - gameovers
    return {
        'network': name,
        'architecture': 'Vanilla DDQN: 16-64-64-64-4',
        'mask_mode': mode,
        'episodes': n,
        'average_score': statistics.mean(scores),
        'median_score': statistics.median(scores),
        'best_score': max(scores),
        'average_max_tile': statistics.mean(tiles),
        'median_max_tile': statistics.median(tiles),
        'average_steps': statistics.mean(steps),
        'median_steps': statistics.median(steps),
        'invalid_terminations': invalid,
        'invalid_termination_pct': 100 * invalid / n,
        'game_over_terminations': gameovers,
        'game_over_termination_pct': 100 * gameovers / n,
        'total_mask_fallbacks': 0,
        'validity_accuracy': '', 'validity_precision': '', 'validity_recall': '',
        'false_invalid_rate': '', 'false_valid_rate': '',
        'validity_TP': 0, 'validity_FP': 0, 'validity_TN': 0, 'validity_FN': 0,
        'step_cap_terminations': capped,
        'step_cap_termination_pct': 100 * capped / n,
    }


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='2048 vanilla DDQN evaluation diagnostics')
    parser.add_argument('--checkpoint-path', type=Path, help='Path to a single checkpoint')
    parser.add_argument('--checkpoint-dir', type=Path, default=PROJECT_DIR / 'artifacts' / 'VANILLA_DDQN' / 'checkpoints')
    parser.add_argument('--networks', nargs='*', help='Checkpoint names, with or without .pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--modes', nargs='+', choices=['none', 'environment'], default=['none', 'environment'])
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'benchmark' / 'ddqn')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('--episodes and --max-steps must be positive')
    if args.checkpoint_path and args.networks:
        parser.error('Use --checkpoint-path or --networks, not both')
    paths = ([args.checkpoint_path.expanduser().resolve()] if args.checkpoint_path
             else sorted(args.checkpoint_dir.expanduser().glob('*.pth')))
    if args.networks:
        wanted = {name if name.endswith('.pth') else name + '.pth' for name in args.networks}
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        parser.error('No checkpoints found. Supply --checkpoint-path or --checkpoint-dir.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; episodes per mode: {args.episodes}')
    summaries = []
    for path in paths:
        model = load_network(path, device)
        for mode in args.modes:
            # Re-seed so each model/mode receives the same initial RNG stream.
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f'\nCheckpoint: {path.name} | mode: {mode}')
            rows = []
            for ep in range(1, args.episodes + 1):
                result = play_episode(model, mode, args.max_steps)
                rows.append({'network': path.name, 'architecture': model.architecture,
                             'mask_mode': mode, 'episode': ep, **result})
                print(f'Episode {ep:4d} | Score: {result["score"]:6d} | '
                      f'Max tile: {result["max_tile"]:4d} | Steps: {result["steps"]:5d} | '
                      f'Ended: {result["termination"]}')
            detail = args.output_dir / f'ddqn_benchmark_{path.stem}_{mode}.csv'
            write_csv(detail, rows)
            summary = summarize(path.name, mode, rows)
            summaries.append(summary)
            print('Summary:')
            for key in ('average_score', 'median_score', 'best_score', 'average_max_tile',
                        'average_steps', 'invalid_termination_pct', 'game_over_termination_pct',
                        'step_cap_termination_pct'):
                print(f'  {key}: {summary[key]:.2f}')
            print(f'Per-episode CSV: {detail}')
    summary_path = args.output_dir / 'benchmark_summary.csv'
    write_csv(summary_path, summaries)
    print(f'\nCombined summary: {summary_path}')

if __name__ == '__main__':
    main()
