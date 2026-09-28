import torch
import math
import game
def state_gen(board_state):
        transposed =  [list(row) for row in zip(*board_state)]
        reduced_board = [
            [math.log2(val) if val!=0 else 0 for val in row ] 
            for row in transposed
        ]
        flattened = torch.flatten(torch.tensor(reduced_board)).tolist()
        return flattened

def board_gen(flattened, rows=4, cols=4):
    reduced_board = [
        flattened[i * cols : (i + 1) * cols] 
        for i in range(rows)
    ]
    transposed = [
        [int(2**val) if val != 0 else 0 for val in row]
        for row in reduced_board
    ]
    original_board_state = [list(row) for row in zip(*transposed)]
    
    return original_board_state

def max_tile(matrix):
    max = matrix[0][0]
    for i in matrix:
        for j in i:
                if j>max:
                    max = j
    return max
def valid_actions(current_state,q_values):
        current_state = current_state.tolist()
        masking_board = game.Board()
        neg_inf = float('-inf')
        valid_action = [neg_inf for _ in range(4)]
        actions = [masking_board.move_up,masking_board.move_left,masking_board.move_right,masking_board.move_down]
        for id,fn in enumerate(actions):
            masking_board.board_state = board_gen(current_state)
            fn()
            next_state = state_gen(masking_board.board_state)
            """
            test_state = board_gen(current_state)
            round_trip = state_gen(test_state)

            print(current_state)
            print(round_trip)
            print(current_state == round_trip)
            """
            if current_state != next_state:
                valid_action[id] = 1
        #print(type(valid_action), valid_action)
        masked_values = [q_values[i] if valid_action[i] == 1 else valid_action[i] for i in range(4)]
        return masked_values