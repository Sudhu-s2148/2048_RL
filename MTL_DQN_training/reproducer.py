import subprocess
import sys
from pathlib import Path
import json

# Assumes reproducer.py and training.py are in the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = SCRIPT_DIR / "training.py"

# Edit these entries to choose which experiments to run.
# Each run has a distinct seed and session name.
RUNS = [
    {"seed": 42, "session": "1.1.1"},
    {"seed": 43, "session": "1.1.2"},
    {"seed": 44, "session": "1.1.3"},
]


def main():
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Could not find training.py here:\n{TRAINING_SCRIPT}\n"
            "Place reproducer.py beside training.py, or update TRAINING_SCRIPT."
        )

    for index, run in enumerate(RUNS, start=1):
        seed = run["seed"]
        session = run["session"]

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


if __name__ == "__main__":
    main()

