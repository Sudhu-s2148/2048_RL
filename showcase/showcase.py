import sys
from pathlib import Path
import os

import pygame
import torch
import math
import random

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
import game

import sys
from pathlib import Path

# Absolute path or relative path to the target directory
target_dir = Path("C:/Users/sudha/Documents/2048_RL/training_environment").resolve()

# Add to sys.path if not already present
if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))

# Import your module directly by filename (omit .py or .pyd)
import agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:/Users/sudha/Documents/2048_RL/artifacts/checkpoints/network_404.pth"

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 650

BOARD_SIZE = 4

CELL_SIZE = 105
CELL_GAP = 12

BOARD_X = 14
BOARD_Y = 150

FPS = 4


# ============================================================
# COLORS
# ============================================================

BACKGROUND = (250, 248, 239)
BOARD_BACKGROUND = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)

DARK_TEXT = (60, 58, 50)
LIGHT_TEXT = (249, 246, 242)


# ============================================================
# TILE COLORS
# ============================================================

TILE_COLORS = {
    0:   (205, 193, 180),
    2:   (238, 228, 218),
    4:   (237, 224, 200),
    8:   (242, 177, 121),
    16:  (245, 149, 99),
    32:  (246, 124, 95),
    64:  (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}


# ============================================================
# STATE REPRESENTATION
# ============================================================

def state_gen(board_state):
    transposed =  [list(row) for row in zip(*board_state)]
    for row in board_state:
        print(row)
    reduced_board = [
        [math.log2(val) if val!=0 else 0 for val in row ] 
        for row in transposed
    ]
    flattened = torch.flatten(torch.tensor(reduced_board)).tolist()
    return flattened


# ============================================================
# DRAW TILE
# ============================================================

def draw_tile(screen, value, row, col, number_font):

    x = BOARD_X + col * (CELL_SIZE + CELL_GAP)
    y = BOARD_Y + row * (CELL_SIZE + CELL_GAP)

    color = TILE_COLORS.get(
        value,
        (60, 58, 50)
    )

    pygame.draw.rect(
        screen,
        color,
        (x, y, CELL_SIZE, CELL_SIZE),
        border_radius=6
    )

    # Don't draw a number for empty cells
    if value == 0:
        return

    # Smaller font for large numbers
    if value >= 1000:
        font = pygame.font.SysFont(
            "arial",
            30,
            bold=True
        )

    elif value >= 100:
        font = pygame.font.SysFont(
            "arial",
            36,
            bold=True
        )

    else:
        font = number_font

    text_color = (
        DARK_TEXT
        if value in [2, 4]
        else LIGHT_TEXT
    )

    text = font.render(
        str(value),
        True,
        text_color
    )

    text_rect = text.get_rect(
        center=(
            x + CELL_SIZE // 2,
            y + CELL_SIZE // 2
        )
    )

    screen.blit(text, text_rect)


# ============================================================
# DRAW ENTIRE BOARD
# ============================================================

def draw_board(
    screen,
    board_state,
    score,
    max_tile,
    title_font,
    score_font,
    number_font
):

    screen.fill(BACKGROUND)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = title_font.render(
        "2048 RL Agent",
        True,
        DARK_TEXT
    )

    screen.blit(
        title,
        (14, 20)
    )


    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score_text = score_font.render(
        f"SCORE\n{score}",
        True,
        DARK_TEXT
    )

    screen.blit(
        score_text,
        (15, 75)
    )


    # --------------------------------------------------------
    # Max tile
    # --------------------------------------------------------

    max_text = score_font.render(
        f"MAX TILE\n{max_tile}",
        True,
        DARK_TEXT
    )

    screen.blit(
        max_text,
        (350, 75)
    )


    # --------------------------------------------------------
    # Board background
    # --------------------------------------------------------

    board_width = (
        BOARD_SIZE * CELL_SIZE
        + (BOARD_SIZE - 1) * CELL_GAP
        + 2 * BOARD_X
    )

    board_height = (
        BOARD_SIZE * CELL_SIZE
        + (BOARD_SIZE - 1) * CELL_GAP
        + 2 * 15
    )

    pygame.draw.rect(
        screen,
        BOARD_BACKGROUND,
        (
            0,
            BOARD_Y - 15,
            board_width,
            board_height
        ),
        border_radius=8
    )


    # --------------------------------------------------------
    # Tiles
    # --------------------------------------------------------

    for row in range(BOARD_SIZE):

        for col in range(BOARD_SIZE):

            value = board_state[row][col]

            draw_tile(
                screen,
                value,
                row,
                col,
                number_font
            )


# ============================================================
# MAIN
# ============================================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    pygame.display.set_caption(
        "2048 RL Showcase"
    )

    clock = pygame.time.Clock()


    # Fonts

    title_font = pygame.font.SysFont(
        "arial",
        42,
        bold=True
    )

    score_font = pygame.font.SysFont(
        "arial",
        22,
        bold=True
    )

    number_font = pygame.font.SysFont(
        "arial",
        42,
        bold=True
    )


    # ========================================================
    # LOAD TRAINED NETWORK
    # ========================================================

    online_network = agent.Agent(
        learning_rate=0.0003,
        weight_decay=1e-4
    ).to(device)

    online_network.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )
    print("Loading from:", os.path.abspath(MODEL_PATH))
    print("File exists:", os.path.exists(MODEL_PATH))

    online_network.eval()


    # ========================================================
    # CREATE NEW GAME
    # ========================================================

    board = game.Board()

    board.spawn_number()

    board.is_game_over()


    running = True


    # ========================================================
    # GAME LOOP
    # ========================================================

    while running:

        # ----------------------------------------------------
        # Pygame events
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False


        # ----------------------------------------------------
        # Make AI move
        # ----------------------------------------------------

        if board.game_state:

            current_state = state_gen(
                board.board_state
            )

            # epsilon = 0
            # Completely greedy / exploitation
            action = online_network.choice(
                current_state,
                0
            )[0]
            print(action)
            '''
            action = random.randint(0,3)'''


            # ------------------------------------------------
            # Execute action
            # ------------------------------------------------

            if action == 0:

                board.move_up()

            elif action == 1:

                board.move_left()

            elif action == 2:

                board.move_right()

            elif action == 3:

                board.move_down()


            # ------------------------------------------------
            # Update game-over state
            # ------------------------------------------------

            board.is_game_over()


        # ----------------------------------------------------
        # Find maximum tile
        # ----------------------------------------------------

        max_tile = max(
            max(row)
            for row in board.board_state
        )


        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------
        state = [list(row) for row in zip(*board.board_state)]
        draw_board(
            screen,
            board.board_state,
            board.score,
            max_tile,
            title_font,
            score_font,
            number_font
        )


        # ----------------------------------------------------
        # Game over message
        # ----------------------------------------------------

        if not board.game_state:

            overlay = pygame.Surface(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.SRCALPHA
            )

            overlay.fill(
                (255, 255, 255, 120)
            )

            screen.blit(
                overlay,
                (0, 0)
            )

            game_over_font = pygame.font.SysFont(
                "arial",
                48,
                bold=True
            )

            game_over_text = game_over_font.render(
                "GAME OVER",
                True,
                DARK_TEXT
            )

            text_rect = game_over_text.get_rect(
                center=(
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT // 2
                )
            )

            screen.blit(
                game_over_text,
                text_rect
            )


        pygame.display.flip()

        clock.tick(FPS)


    pygame.quit()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
