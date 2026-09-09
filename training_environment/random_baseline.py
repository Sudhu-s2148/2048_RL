import game
import math
import random
import json
from itertools import islice

total_episodes = 100
data = {}
data["format"] = ["total_score","max_tile","ep_length"]

def max_tile(matrix):
    max = matrix[0][0]
    for i in matrix:
        for j in i:
                if j>max:
                    max = j
    return max

#running random testing to set baseline
for episode in range(total_episodes):
    board = game.Board()
    board.spawn_number()
    ep_length = 0
    count = 0
    while board.game_state!=False:
        for i in range(4):
            transposed =  [list(row) for row in zip(*board.board_state)]
            reduced_board = [
                [math.log2(val) if val!=0 else 0 for val in row ] 
                for row in transposed
            ]
        choice  = random.randint(0,3)
        prev_score = board.score 
        if choice == 0:
            board.move_up()           
        elif choice == 1:
                board.move_left()                
        elif choice == 2:
                board.move_right()                
        elif choice == 3:
                board.move_down()
        else:
            break

        count+=1
        board.is_game_over()
    data[episode] = [board.score,max_tile(board.board_state),count]

#calculating the avg
total_score = 0
total_steps = 0
total_max_tile = 0
for key, value in islice(data.items(), 1, None):
    total_score += value[0]
    total_max_tile += value[1]
    total_steps += value[2]
avg_score = total_score/total_episodes
avg_ep_length = total_steps/total_episodes
avg_max_tile = total_max_tile/total_episodes
    
data["average_score"] = avg_score
data["average_episode_length"] = avg_ep_length
data["average_max_tile"] = avg_max_tile

#storing them for future purposes
with open("output_data/random_100.json","w",encoding="utf-8") as fh:
     json.dump(data,fh,indent = 4)

for i in data:
    print(data[i])