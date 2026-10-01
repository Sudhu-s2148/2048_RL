import sys
import asyncio
import sys
from pathlib import Path


# This line stays EXACTLY as it is:


project_dir = Path(__file__).resolve().parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Replace 'my_module' with your actual .pyd filename (no extension):
import game
import torch
import agent, buffer
import copy
import math
import csv,json
from helpers import state_gen, max_tile


#########################################################################
async def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    bot_script_path = project_dir / "discord_manager" / "bot.py"  # Ensure this path matches your bot file location
    bot_process = await asyncio.create_subprocess_exec(
        sys.executable, str(bot_script_path)
    )
    print("Discord bot running in background...")

    session = 10

    learning_rate = 0.0003
    weight_decay = 1e-4

    episode = 0
    total_steps = 0
    target_sync_count = 0

    total_episodes = 10000
    batch_size = 64
    target_sync = 1000

    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.09
    epsilon_decay = epsilon_min**(1/(.89*total_episodes))

    best_tile = 0
    best_score = 0

    #########################################################################
    data = {}
    ep_data = {}
    data["parameters"] = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "batch_size": batch_size,
        "replay_buffer_size": 10000,
        "target_sync": target_sync,
        "epsilon_start": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min
    }
    data["format"] = [
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
        "consecutive_invalid_moves",
        "terminated_by_invalid_cap",
        "max_consecutive_invalid_moves",
        "epsilon"
    ]

    data["episodes"] = []

    save_path_network = project_dir / "artifacts" / "checkpoints" / f"network_{session}.pth"
    save_path_csv = project_dir / "artifacts" / "output_data" / f"run_{session}.csv"
    save_path_json = project_dir / "artifacts" / "output_data" / f"run_{session}.json"
    save_path_discord = project_dir / "discord_manager" / "status.json"

    #setting up the network and other variables
    online_network = agent.Agent(learning_rate,weight_decay).to(device)
    replay_buffer = buffer.exp_buffer(10000)
    target_network = copy.deepcopy(online_network).to(device)

    #to genarate the flattened state with each tile value being the power of 2
    #now in helpers.py
    ##########################################################################
    
    for episode in range(total_episodes):
        board = game.Board()
        board.spawn_number()

        #telemetry variables reset
        steps = 0

        greatest_tile_ever = 0
        valid_moves = 0
        invalid_moves = 0
        merging_moves = 0
        non_merging_valid_moves = 0

        random_invalid_moves = 0
        random_valid_moves = 0

        exploit_invalid_moves =0
        exploit_valid_moves = 0

        consecutive_invalid_moves = 0
        terminated_by_invalid_cap = 0
        max_consecutive_invalid_moves = 0

        #game loop
        done = board.game_state
        while done!=False:

            current_state = state_gen(board.board_state)
            prev_score = board.score 

            #action choice
            action,type  = online_network.choice(current_state,epsilon)
            #print("action:",action)
            if action == 0:
                board.move_up()           
            elif action == 1:
                    board.move_left()                
            elif action == 2:
                    board.move_right()                
            elif action == 3:
                    board.move_down()

            #reward calc
            next_state = state_gen(board.board_state)
            current_score = board.score
            reward = current_score-prev_score
            if reward!=0:
                merging_moves+=1
            if current_state == next_state:
                #print(current_state,next_state)
                #print("invalid move")
                reward-=1
                invalid_moves+=1
                '''consecutive_invalid_moves+=1
                if consecutive_invalid_moves>max_consecutive_invalid_moves:
                     max_consecutive_invalid_moves=consecutive_invalid_moves'''
                if type == 1:
                    exploit_invalid_moves+=1
                else:
                    random_invalid_moves+=1
            else:
                #consecutive_invalid_moves = 0
                valid_moves+=1
                if type == 1:
                    exploit_valid_moves+=1
                else:
                    random_valid_moves+=1

                if reward == 0:
                    non_merging_valid_moves+=1

            board.is_game_over()

            done = board.game_state
            '''if consecutive_invalid_moves >= 10:
                 terminated_by_invalid_cap = 1
                 done = True'''

            #buffer update
            exp = [current_state,action,next_state,reward,done]
            replay_buffer.append(exp)

            steps+=1

            #updating weights
            if replay_buffer.len()>=batch_size:
                    train_batch = replay_buffer.sample(batch_size)
                    '''print(
                        f"\nEpisode: {episode} | Step: {steps} | "
                        f"Buffer: {len(replay_buffer.buffer)} | "
                        f"State shape: {len(train_batch[0][0])} | "
                        f"Action: {train_batch[0][1]} | "
                        f"Reward: {train_batch[0][3]} | "
                        f"Done: {train_batch[0][4]}"
                    )'''
                    online_network.update(train_batch,target_network,gamma)

            target_sync_count+=1

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        best_tile = max_tile(board.board_state)

        print(
            f"Episode {episode + 1} | "
            f"Score: {board.score} | "
            f"Max tile: {best_tile} | "
            f"Steps: {steps} | "
            f"valid moves: {valid_moves}| "
            f"invalid moves:{invalid_moves}| "
            f"merging moves: {merging_moves}| "
            f"Epsilon: {epsilon:.4f}| "

        )

        #training data
        if greatest_tile_ever<best_tile:
             greatest_tile_ever = best_tile
        data["episodes"].append([
            episode,
            board.score,
            best_tile,
            steps,
            valid_moves,
            invalid_moves,
            merging_moves,
            non_merging_valid_moves,
            random_invalid_moves,
            random_valid_moves,    
            exploit_invalid_moves,
            exploit_valid_moves,
            consecutive_invalid_moves,
            terminated_by_invalid_cap,
            max_consecutive_invalid_moves,
            epsilon
        ])


        #discord data
        if best_score<board.score:
                best_score = board.score
        if best_tile<max_tile(board.board_state):
            best_tile = max_tile(board.board_state)

        ep_data["session"] = session
        ep_data["episode"] = episode+1
        ep_data["total_episodes"] = total_episodes
        ep_data["epsilon"] = epsilon
        ep_data["score"] = board.score
        ep_data["best_score"] = best_score
        ep_data["best_tile"] = best_tile
        ep_data["ep_length"] = steps
        ep_data["valid_moves"] = valid_moves
        ep_data["invalid_moves"] = invalid_moves
        ep_data["merging_moves"] = merging_moves
        ep_data["random_invalid_moves"] = random_invalid_moves
        ep_data["exploit_invalid_moves"] = exploit_invalid_moves

        with open(save_path_discord, "w") as file:
            json.dump(ep_data, file, indent=4)


        #syncing the target network with online network
        if target_sync_count == target_sync:
            target_network = copy.deepcopy(online_network)
            target_sync_count = 0

    data["best_tile"] = greatest_tile_ever
    data["best_score"] = best_score

    print("best_tile:",greatest_tile_ever)
    print("best_score",best_score)

    #saving final network,json,csv data files
    torch.save(online_network.state_dict(),save_path_network)

    with open(save_path_json, "w") as file:
        json.dump(data, file, indent=4)

    with open(save_path_csv, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(data["format"])

        for episode_data in data["episodes"]:
            writer.writerow(episode_data)

    #cleaningly closing discord process
    try:
        await asyncio.wait_for(bot_process.wait(), timeout=30)
        print("Discord bot shut down gracefully. Training pipeline complete.")
    except asyncio.TimeoutError:
        print("Discord bot did not shut down within 30 seconds. Terminating it...")
        bot_process.terminate()
        try:
            await asyncio.wait_for(bot_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            bot_process.kill()
            await bot_process.wait()
        print("Discord bot terminated. Training pipeline complete.")

if __name__ == "__main__":
    asyncio.run(main())
