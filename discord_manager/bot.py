import time
import os
import sys
import json
import glob
import asyncio
import discord
from discord.ext import tasks
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFICATION_CHANNEL_ID = 1547985485954027723 

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BOT_DIR, ".."))
STATUS_FILE = os.path.join(BOT_DIR, "status.json")
LOCK_FILE = os.path.join(BOT_DIR, "bot.lock")

training_finished_notified = False


def claim_completion_lock():
    """Atomically create LOCK_FILE. Returns True if this process won the
    race and should handle completion, False if another process already
    claimed it. os.O_CREAT|O_EXCL is atomic at the OS level, so this is
    safe even if two bot.py processes hit this at the same instant."""
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


async def run_analysis_and_send(channel):
    """Executes analyze.py located in the parent directory using status.json."""
    script_path = os.path.join(PARENT_DIR, "analyze.py")
    analysis_dir = os.path.join(PARENT_DIR, "analysis")

    try:
        with open(STATUS_FILE, "r") as file:
            status = json.load(file)
        session = status.get("session")
    except (json.JSONDecodeError, OSError, KeyError) as e:
        await channel.send(f"❌ Failed to read `status.json`: {e}")
        return

    json_path = os.path.join(PARENT_DIR, "artifacts", "output_data", f"run_{session}.json")

    for old_file in glob.glob(os.path.join(analysis_dir, "*.png")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    process = await asyncio.create_subprocess_exec(
        sys.executable, script_path, json_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PARENT_DIR
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        err_msg = stderr.decode().strip()
        if len(err_msg) > 1800:
            err_msg = err_msg[-1800:]
            
        await channel.send(f"❌ Error running `analyze.py`:\n```\n...{err_msg}\n```")
        return

    png_files = glob.glob(os.path.join(analysis_dir, "*.png"))

    if png_files:
        files_to_send = [discord.File(filepath) for filepath in png_files[:10]]
        await channel.send("📈 **Training Analysis Plots:**", files=files_to_send)
        for file in files_to_send:
            file.close()
    else:
        await channel.send("⚠️ `analyze.py` executed successfully, but no plot files were found in `./analysis/`.")


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    # training.py spawns a fresh bot.py process for every run, so any
    # status.json / bot.lock left over from a *previous* run is stale by
    # definition. If we don't clear them, check_status can read yesterday's
    # "finished" status on its very first tick, or find an old lock file
    # and refuse to ever send this run's message.
    for stale_file in (STATUS_FILE, LOCK_FILE):
        if os.path.exists(stale_file):
            try:
                os.remove(stale_file)
            except OSError:
                pass

    if not check_status.is_running():
        check_status.start()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.strip().lower() != "!status":
        return

    if not os.path.exists(STATUS_FILE):
        await message.channel.send("⚠️ No `status.json` found yet — training may not have started.")
        return

    try:
        with open(STATUS_FILE, "r") as file:
            status = json.load(file)
    except (json.JSONDecodeError, OSError) as e:
        await message.channel.send(f"❌ Failed to read `status.json`: {e}")
        return

    episode = status.get("episode", "?")
    total_episodes = status.get("total_episodes", "?")
    await message.channel.send(
        f"📊 **Training Status**\n"
        f"Episode: {status['episode']} / {status['total_episodes']}\n"
        f"Current score: {status['score']}\n"
        f"Best score: {status['best_score']}\n"
        f"Best tile: {status['best_tile']}\n"
        f"\n"
        f"total moves:{status['ep_length']}\n"
        f"valid moves: {status['valid_moves']}\n"
        f"invalid moves: {status['invalid_moves']}\n"
        f"random invalid moves: {status['random_invalid_moves']}\n"
        f"exploit invalid moves: {status['exploit_invalid_moves']}\n"
        f"merging moves: {status['merging_moves']}\n"
        f"\n"
        f"Epsilon: {status['epsilon']:.3f}"
        f"\n"
    )


@tasks.loop(seconds=5)
async def check_status():
    global training_finished_notified

    # Once we've handled completion for this run, do nothing else.
    if training_finished_notified:
        return

    if not os.path.exists(STATUS_FILE):
        return

    try:
        with open(STATUS_FILE, "r") as file:
            status = json.load(file)

        episode = status.get("episode", 0)
        total_episodes = status.get("total_episodes", 1)

        if episode >= total_episodes:
            # Guard against a second, genuinely concurrent bot.py process
            # (e.g. a stray instance left running from earlier testing)
            # also seeing completion at the same time. Only the process
            # that wins the atomic lock proceeds; the loser stops quietly
            # without touching the plot files or sending anything.
            if not claim_completion_lock():
                print("Another bot process already claimed completion. Exiting quietly.")
                check_status.stop()
                await client.close()
                return

            training_finished_notified = True
            check_status.stop()

            channel = client.get_channel(NOTIFICATION_CHANNEL_ID)
            if channel:
                await channel.send(
                    f"🎉 **Training Complete!**\n"
                    f"Final Episode: {status['episode']} / {status['total_episodes']}\n"
                    f"Final Score: {status['score']}\n"
                    f"Best Score: {status['best_score']}\n"
                    f"Best Tile: {status['best_tile']}"
                )
                
                await run_analysis_and_send(channel)

            print("Final completion message and plots sent. Shutting down bot process...")
            await client.close()

    except (json.JSONDecodeError, OSError):
        pass


if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except Exception as e:
        print(f"Bot exited: {e}")

