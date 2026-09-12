import sys
from pathlib import Path

# This line stays EXACTLY as it is:
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))



# Replace 'my_module' with your actual .pyd filename (no extension):
import game
import torch
import agent, buffer
import copy
import math
import csv,json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#########################################################################
session = 404

learning_rate = 0.0003
weight_decay = 1e-4

episode = 0
total_steps = 0
target_sync_count = 0

total_episodes = 5000
batch_size = 64
target_sync = 1000

gamma = 0.99
epsilon = 1
epsilon_decay = 0.99943
epsilon_min = 0.09

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
    "replay_buffer_size": batch_size,
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
    "epsilon"
]

data["episodes"] = []

save_path_network = f"C:/Users/sudha/Documents/2048_RL/artifacts/checkpoints/network_{session}.pth"
save_path_csv = f"C:/Users/sudha/Documents/2048_RL/artifacts/output_data/run_{session}.csv"
save_path_json = f"C:/Users/sudha/Documents/2048_RL/artifacts/output_data/run_{session}.json"
save_path_discord = f"C:/Users/sudha/Documents/2048_RL/discord_manager/status.json"
#setting up the network and other variables
online_network = agent.Agent(learning_rate,weight_decay).to(device)
replay_buffer = buffer.exp_buffer(10000)
target_network = copy.deepcopy(online_network).to(device)

#to genarate the flattened state with each tile value being the power of 2
def state_gen(board_state):
    transposed =  [list(row) for row in zip(*board_state)]
    reduced_board = [
        [math.log2(val) if val!=0 else 0 for val in row ] 
        for row in transposed
    ]
    flattened = torch.flatten(torch.tensor(reduced_board)).tolist()
    return flattened

def max_tile(matrix):
    max = matrix[0][0]
    for i in matrix:
        for j in i:
                if j>max:
                    max = j
    return max

##########################################################################

for episode in range(total_episodes):
    board = game.Board()
    board.spawn_number()
    steps = 0
    valid_moves = 0
    invalid_moves = 0
    merging_moves = 0
    non_merging_valid_moves = 0
    done = board.game_state
    while done!=False:

        current_state = state_gen(board.board_state)
        prev_score = board.score 
        action  = online_network.choice(current_state,epsilon)

        if action == 0:
            board.move_up()           
        elif action == 1:
                board.move_left()                
        elif action == 2:
                board.move_right()                
        elif action == 3:
                board.move_down()

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
        else:
             valid_moves+=1
             if reward == 0:
                  non_merging_valid_moves+=1

        
        board.is_game_over()
        done = board.game_state

        exp = [current_state,action,next_state,reward,done]
        replay_buffer.append(exp)

        steps+=1

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
        f"Epsilon: {epsilon:.4f}|"
        f"valid moves: {valid_moves}|"
        f"merging moves: {merging_moves}"

    )

    data["episodes"].append([
        episode,
        board.score,
        best_tile,
        steps,
        valid_moves,
        invalid_moves,
        merging_moves,
        non_merging_valid_moves,
        epsilon
    ])

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
    ep_data["merging_moves"] = merging_moves

    with open(save_path_discord, "w") as file:
        json.dump(ep_data, file, indent=4)



    if target_sync_count == target_sync:
        target_network = copy.deepcopy(online_network)
        target_sync_count = 0

data["best_tile"] = best_tile
data["best_score"] = best_score

print("best_tile:",best_tile)
print("best_score",best_score)

torch.save(online_network.state_dict(),save_path_network)

with open(save_path_json, "w") as file:
    json.dump(data, file, indent=4)

with open(save_path_csv, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(data["format"])

    for episode_data in data["episodes"]:
        writer.writerow(episode_data)
