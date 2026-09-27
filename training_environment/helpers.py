import torch
import math
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