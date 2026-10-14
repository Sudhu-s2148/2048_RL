import subprocess
import sys
from pathlib import Path
import json
import argparse

# Assumes reproducer.py and training.py are in the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = SCRIPT_DIR / "training.py"
BENCHMARKIN_SCRIPT = SCRIPT_DIR.parent/"benchmark"/"benchmark_mtl.py"

# Edit these entries to choose which experiments to run.
# Each run has a distinct seed and session name.
RUNS = [
    {"seed": 44, "session": "1.1.3.2"},
]

parser = argparse.ArgumentParser()
parser.add_argument("--total_episodes",type=int,default = 10)
args = parser.parse_args()

def main():
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Could not find training.py here:\n{TRAINING_SCRIPT}\n"
            "Place reproducer.py beside training.py, or update TRAINING_SCRIPT."
        )

    for index, run in enumerate(RUNS, start=1):
        seed = run["seed"]
        session = run["session"]
        total_episodes = args.total_episodes

        data = {}
        data["current_run"] = index
        data["total_runs"] = len(RUNS)
        with open("run.json","w") as file:
                    json.dump(data,file)
        

        command = [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--seed", str(seed),
            "--session", str(session),
            "--total_episodes", str(total_episodes),
        ]

        print("\n" + "=" * 72, flush=True)
        print(
            f"Starting run {index}/{len(RUNS)} | seed={seed} | session={session}",
            flush=True,
        )
        print("Command:", subprocess.list2cmdline(command), flush=True)
        print("=" * 72, flush=True)

        # Runs from the training script's directory, so relative file paths
        # inside training.py continue to resolve from that folder.
        result = subprocess.run(command, cwd=SCRIPT_DIR)

        if result.returncode != 0:
            print(
                f"\nRun failed (exit code {result.returncode}). "
                "Stopping; later runs were not started.",
                flush=True,
            )
            raise SystemExit(result.returncode)

        print(
            f"\nCompleted run {index}/{len(RUNS)} | "
            f"seed={seed} | session={session}",
            flush=True,
        )

    print("\nAll configured runs completed successfully.", flush=True)

    networks = [str("network_"+i["session"]) for i in RUNS]
    benchmark_command = [
        sys.executable,
        str(BENCHMARKIN_SCRIPT),
        "--networks"
    ]+networks

    bench_result = subprocess.run(benchmark_command, cwd=SCRIPT_DIR.parent/"benchmark")

    if bench_result.returncode != 0:
                print(
                    f"\nRun failed (exit code {bench_result.returncode}). "
                    "Stopping; later runs were not started.",
                    flush=True,
                )
                raise SystemExit(bench_result.returncode)

if __name__ == "__main__":
    main()

