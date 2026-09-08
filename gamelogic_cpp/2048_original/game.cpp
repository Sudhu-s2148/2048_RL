#include <iostream>
#include <string>
#include <random>
#include <cmath>

#include "raylib.h"

struct iVec2{
    int x;
    int y;
};

class Board
{
    private:
        int width;
        int height;
        Color color;
        std::vector<float>grid_params = std::vector<float>(4); 
        static std::mt19937 m_gen;
        

    public:
        std::vector<std::vector<int>> board_state = std::vector<std::vector<int>>(4, std::vector<int>(4, 0));
        bool game_state;
        int score;

        Board(int w, int h, Color c)
        {
            width = w;
            height = h;
            color = c;
            game_state = true;
            score = 0;
        }

        int get_random(int min, int max)
        {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(m_gen);
        }

        void init_grid()
        {
            float x_start = width/4.0f;
            float y_start = height/4.0f;

            float row_width = height/8.0f;
            float column_width = width/8.0f;

            grid_params = {x_start, y_start, row_width, column_width};
        }

        void draw_grid()
        {
            for (int i = 0; i <= 4; i++)
            {
                float x_start = grid_params[0] + i * grid_params[3];
                float y_end = grid_params[1] + 4 * grid_params[2];
                DrawLineV({x_start, grid_params[1]}, {x_start, y_end}, color);
            }

            for (int i = 0; i <= 4; i++)
            {
                float y_start = grid_params[1] + i * grid_params[2];
                float x_end = grid_params[0] + 4 * grid_params[3];
                DrawLineV({grid_params[0], y_start}, {x_end, y_start}, color);
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (board_state[i][j] != 0)
                    {
                        std::string num_str = std::to_string(board_state[i][j]);
                        int fontSize = 60;
                        int textWidth = MeasureText(num_str.c_str(), fontSize);
                        
                        float max_width = grid_params[3] - 20; 
                        
                        if (textWidth > max_width)
                        {
                            fontSize = (int)((fontSize * max_width) / textWidth);
                            textWidth = MeasureText(num_str.c_str(), fontSize);
                        }

                        float cell_x = grid_params[0] + i * grid_params[3];
                        float cell_y = grid_params[1] + j * grid_params[2];
                        
                        float text_x = cell_x + (grid_params[3] / 2.0f) - (textWidth / 2.0f);
                        float text_y = cell_y + (grid_params[2] / 2.0f) - (fontSize / 2.0f);

                        float power = std::log2(board_state[i][j]);
                        float lerp_amount = power/11.0f;
                        if (lerp_amount > 1.0f) lerp_amount = 1.0f;

                        Color draw_color = ColorLerp(LIGHTGRAY, MAROON, lerp_amount);

                        DrawText(num_str.c_str(), text_x, text_y, fontSize, draw_color);
                    }
                }
            }

            DrawText(std::to_string(score).c_str(), width/2, height - 40, 20, WHITE);
        }

        void spawn_number()
        {
            std::vector<iVec2> free_spaces;

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (board_state[i][j] == 0)
                    {
                        free_spaces.push_back({i, j});
                    }
                }
            }

            if (free_spaces.size())
            {
                iVec2 spawn_coord = free_spaces[get_random(0, free_spaces.size() - 1)];
                /* std::cout << spawn_coord.x << ", " << spawn_coord.y << std::endl; */

                int spawn_prob = get_random(0, 100);

                int spawn_number;

                if (spawn_prob >= 90)
                {
                    spawn_number = 4;
                }
                else 
                {
                    spawn_number = 2;
                }

                board_state[spawn_coord.x][spawn_coord.y] = spawn_number;
            }
            else 
            {
                std::cout << "haha loser" << std::endl;
            }  
        }
        
        void move_left()
        {
            bool board_changed = false;

            for (int j = 0; j < 4; j++)
            {
                int insert_pos = 0;
                bool previous_merged = false;

                for (int i = 0; i < 4; i++)
                {
                    if (board_state[i][j] != 0)
                    {
                        int curr_value = board_state[i][j];
                        board_state[i][j] = 0;

                        if (insert_pos > 0 && board_state[insert_pos - 1][j] == curr_value)
                        {
                            board_state[insert_pos - 1][j] *= 2;
                            score += board_state[insert_pos - 1][j];
                            previous_merged = true;
                            board_changed = true;
                        }
                        else 
                        {
                            board_state[insert_pos][j] = curr_value;
                            if (insert_pos != i)
                            {
                                board_changed = true;
                            }

                            insert_pos++;
                            previous_merged = false;
                        }
                    }
                }
            }
            
            if (board_changed)
            {
                spawn_number();
            }
        }

        void move_right()
        {
            bool board_changed = false;

            for (int j = 0; j < 4; j++)
            {
                int insert_pos = 3;
                bool previous_merged = false;

                for (int i = 3; i >=0 ; i--)
                {
                    if (board_state[i][j] != 0)
                    {
                        int curr_value = board_state[i][j];
                        board_state[i][j] = 0;

                        if (insert_pos < 3 && board_state[insert_pos + 1][j] == curr_value)
                        {
                            board_state[insert_pos + 1][j] *= 2;
                            score += board_state[insert_pos + 1][j];
                            previous_merged = true;
                            board_changed = true;
                        }
                        else 
                        {
                            board_state[insert_pos][j] = curr_value;
                            if (insert_pos != i)
                            {
                                board_changed = true;
                            }

                            insert_pos--;
                            previous_merged = false;
                        }
                    }
                }
            }
            
            if (board_changed)
            {
                spawn_number();
            }
        }

        void move_up()
        {
            bool board_changed = false;

            for (int i = 0; i < 4; i++)
            {
                int insert_pos = 0;
                bool previous_merged = false;

                for (int j = 0; j < 4; j++)
                {
                    if (board_state[i][j] != 0)
                    {
                        int curr_value = board_state[i][j];
                        board_state[i][j] = 0;

                        if (insert_pos > 0 && board_state[i][insert_pos - 1] == curr_value)
                        {
                            board_state[i][insert_pos - 1] *= 2;
                            score += board_state[i][insert_pos - 1];
                            previous_merged = true;
                            board_changed = true;
                        }
                        else 
                        {
                            board_state[i][insert_pos] = curr_value;
                            if (insert_pos != j)
                            {
                                board_changed = true;
                            }

                            insert_pos++;
                            previous_merged = false;
                        }
                    }
                }
            }
            
            if (board_changed)
            {
                spawn_number();
            }
        }

        void move_down()
        {
            bool board_changed = false;

            for (int i = 0; i < 4; i++)
            {
                int insert_pos = 3;
                bool previous_merged = false;

                for (int j = 3; j >= 0; j--)
                {
                    if (board_state[i][j] != 0)
                    {
                        int curr_value = board_state[i][j];
                        board_state[i][j] = 0;

                        if (insert_pos < 3 && board_state[i][insert_pos + 1] == curr_value)
                        {
                            board_state[i][insert_pos + 1] *= 2;
                            score += board_state[i][insert_pos + 1];
                            previous_merged = true;
                            board_changed = true;
                        }
                        else 
                        {
                            board_state[i][insert_pos] = curr_value;
                            if (insert_pos != j)
                            {
                                board_changed = true;
                            }

                            insert_pos--;
                            previous_merged = false;
                        }
                    }
                }
            }
            
            if (board_changed)
            {
                spawn_number();
            }
        }

        void is_game_over()
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (board_state[i][j] == 0)
                    {
                        return;
                    }
                    if (i < 3 && board_state[i][j] == board_state[i+1][j])
                    {
                        return;
                    }
                    if (j < 3 && board_state[i][j] == board_state[i][j+1])
                    {
                        return;
                    }
                }
            }
            
            game_state = false;
            std::cout << "haha loser" << std::endl;
        }

        void reset_board()
        {
            game_state = true;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    board_state[i][j] = 0;
                }
            }
            score = 0;
            spawn_number();
        }

};


std::mt19937 Board::m_gen(std::random_device{}());

int main()
{
    constexpr int WIDTH = 800;
    constexpr int HEIGHT = 800;

    InitWindow(WIDTH, HEIGHT, "test");
    SetTargetFPS(60);

    Board b = Board(WIDTH, HEIGHT, GRAY);
    b.init_grid();
    b.spawn_number();

    while (!WindowShouldClose())
    {
        if (b.game_state == false)
        {
            DrawText("game over. press r to reset or q to quit", WIDTH/2 - 400, 20, 30, WHITE);
            if (IsKeyPressed(KEY_R)) b.reset_board();
            else if (IsKeyPressed(KEY_Q)) exit(EXIT_SUCCESS);
        }
        else 
        {
            if (IsKeyPressed(KEY_A))
            {
                b.move_left();
            }
            else if (IsKeyPressed(KEY_D))
            {
                b.move_right();
            }
            else if(IsKeyPressed(KEY_W))
            {
                b.move_up();
            }
            else if (IsKeyPressed(KEY_S))
            {
                b.move_down();
            }

            b.is_game_over();
        }       

        BeginDrawing();
        ClearBackground(BLACK);
        b.draw_grid();
        EndDrawing();
    }

    CloseWindow();
    return 0;
}