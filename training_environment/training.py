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
import json

#########################################################################
data = {}
data["format"] = ["total_score","max_tile","ep_length"]

session = 0

learning_rate = 0.0003
weight_decay = 1e-4

episode = 0
total_steps = 0
target_sync_count = 0

total_episodes = 1000
batch_size = 64
target_sync = 500

gamma = 0.99
epsilon = 1
decay = 0.99971
epsilon_min = 0.09

#########################################################################

save_path = f"C:/Users/sudha/Documents/2048_RL/checkpoints/network_{session}.pth"

#setting up the network and other variables
online_network = agent.Agent(learning_rate,weight_decay)
replay_buffer = buffer.exp_buffer(10000)
target_network = copy.deepcopy(online_network)

#to genarate the flattened state with each tile value being the power of 2
def state_gen(board_state):
    transposed =  [list(row) for row in zip(*board.board_state)]
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
        board.is_game_over()
        done = board.game_state

        exp = [current_state,action,next_state,reward,done]
        replay_buffer.append(exp)

        steps+=1

        if replay_buffer.len()>=batch_size:
                train_batch = replay_buffer.sample(batch_size)
                print(
                    f"\nEpisode: {episode} | Step: {steps} | "
                    f"Buffer: {len(replay_buffer.buffer)} | "
                    f"State shape: {len(train_batch[0][0])} | "
                    f"Action: {train_batch[0][1]} | "
                    f"Reward: {train_batch[0][3]} | "
                    f"Done: {train_batch[0][4]}"
                )
                online_network.update(train_batch,target_network,gamma)

        target_sync_count+=1

    epsilon = max(epsilon * decay, epsilon_min)
    print(
        f"Episode {episode + 1} | "
        f"Score: {board.score} | "
        f"Max tile: {max_tile(board.board_state)} | "
        f"Steps: {steps} | "
        f"Epsilon: {epsilon:.4f}"
    )
    data[episode] = [board.score,max_tile(board.board_state),steps]

    
    if target_sync_count == target_sync:
        target_network = copy.deepcopy(online_network)
        target_sync_count = 0

torch.save(online_network.state_dict(),save_path)
