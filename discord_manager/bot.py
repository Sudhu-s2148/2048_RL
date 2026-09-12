import os
import json
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

training_finished_notified = False


import glob

import os
import json
import glob
import asyncio
import discord

async def run_analysis_and_send(channel):
    """Executes analyze.py located in the parent directory using local status.json."""
    
    # Define the parent directory (..\) where analyze.py and artifacts live
    parent_dir = os.path.abspath("..")
    script_path = os.path.join(parent_dir, "analyze.py")
    analysis_dir = os.path.join(parent_dir, "analysis")

    # 1. Read status.json locally from current directory (discord_manager)
    try:
        with open("status.json", "r") as file:
            status = json.load(file)
        session = status.get("session")
    except (json.JSONDecodeError, OSError, KeyError) as e:
        await channel.send(f"❌ Failed to read local `status.json`: {e}")
        return

    json_path = os.path.join(parent_dir, "artifacts", "output_data", f"run_{session}.json")

    # 2. Clear old plots in the parent analysis folder
    for old_file in glob.glob(os.path.join(analysis_dir, "*.png")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    # 3. Execute analyze.py with cwd set to the parent directory
    process = await asyncio.create_subprocess_exec(
        "py", script_path, json_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=parent_dir  # Ensures analyze.py runs relative to 2048_RL root
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        await channel.send(f"❌ Error running `analyze.py`:\n```{stderr.decode()}```")
        return

    # 4. Gather generated PNGs from parent's analysis folder
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
    check_status.start()


@tasks.loop(seconds=10)
async def check_status():
    global training_finished_notified

    if not os.path.exists("status.json"):
        return

    try:
        with open("status.json", "r") as file:
            status = json.load(file)

        episode = status.get("episode", 0)
        total_episodes = status.get("total_episodes", 1)

        if episode >= total_episodes and not training_finished_notified:
            channel = client.get_channel(NOTIFICATION_CHANNEL_ID)
            if channel:
                await channel.send(
                    f"🎉 **Training Complete!**\n"
                    f"Final Episode: {status['episode']} / {status['total_episodes']}\n"
                    f"Final Score: {status['score']}\n"
                    f"Best Score: {status['best_score']}\n"
                    f"Best Tile: {status['best_tile']}"
                )
                training_finished_notified = True
                
                # Automatically run and send analysis when training completes
                await run_analysis_and_send(channel)

        elif episode < total_episodes:
            training_finished_notified = False

    except (json.JSONDecodeError, OSError):
        pass


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "!ping":
        await message.channel.send("Pong!")

    elif message.content == "!status":
        if not os.path.exists("status.json"):
            await message.channel.send("No status file found.")
            return

        with open("status.json", "r") as file:
            status = json.load(file)

        await message.channel.send(
            f"📊 **Training Status**\n"
            f"Episode: {status['episode']} / {status['total_episodes']}\n"
            f"Epsilon: {status['epsilon']:.3f}\n"
            f"Current score: {status['score']}\n"
            f"Best score: {status['best_score']}\n"
            f"Best tile: {status['best_tile']}\n"
            f"valid moves: {status['valid_moves']}\n"
            f"episode length(moves): {status['ep_length']}\n"
            f"merging moves: {status['merging_moves']}"
        )

    # Added !analyze command to manually generate and fetch the plot
    elif message.content == "!analyze":
        await message.channel.send("Generating plot from `analyze.py`...")
        await run_analysis_and_send(message.channel)


client.run(TOKEN)