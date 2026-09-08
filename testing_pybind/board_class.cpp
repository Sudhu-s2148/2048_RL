#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class Board
{
public:
    std::vector<std::vector<int>> board_state;
    bool game_state;
    int score;

    Board(int initial_score)
    {
        board_state = {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        };

        game_state = true;
        score = initial_score;
    }

    void add_score(int amount)
    {
        score += amount;
    }

    void set_board(std::vector<std::vector<int>> new_board)
    {
        board_state = new_board;
    }
    
};



namespace py = pybind11;

PYBIND11_MODULE(test, m)
{
    py::class_<Board>(m, "Board")
        .def(py::init<int>())
        .def_readwrite("score",&Board::score)
        .def_readwrite("board",&Board::board_state)
        .def("add_score",&Board::add_score)
        .def("set_board",&Board::set_board);
}