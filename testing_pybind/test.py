import test


b = test.Board(100)
new_board = [
    [2, 0, 4, 0],
    [0, 8, 0, 16],
    [2, 0, 0, 0],
    [0, 0, 0, 4]
]
b.set_board(new_board)
print(b)
b.score = 100
print(b.score)
b.add_score(200)
print(b.score)
print("board",b.board)