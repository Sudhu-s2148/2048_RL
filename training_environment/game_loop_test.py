import game
import math

board = game.Board()
board.spawn_number()
prev_score = 0
while board.game_state!=False:
    for i in range(4):
        transposed =  [list(row) for row in zip(*board.board_state)]
        reduced_board = [
            [math.log2(val) if val!=0 else 0 for val in row ] 
            for row in transposed
        ]
        print(reduced_board[i],"\n")
    choice  = input("enter your move:")
    prev_score = board.score 
    if choice == 'w':
        board.move_up()
        board.is_game_over()
    elif choice == 'a':
            board.move_left()
            board.is_game_over()
    elif choice == 'd':
            board.move_right()
            board.is_game_over()
    elif choice == 's':
            board.move_down()
            board.is_game_over()
    else:
          break
    current_score = board.score
    print("total score:",current_score)
    print("gained in current move:",current_score-prev_score)
    print("----------------")