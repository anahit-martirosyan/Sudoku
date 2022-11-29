from sudoku import Sudoku


# Temporary - for testing
def get_input(file):
    return [
        [1, None, None, None, None, None, None, None, 2],
        [None, None, 8, None, None, 9, None, 3, 7],
        [7, None, None, 5, 3, None, None, 8, None],
        [None, 8, None, None, 7, 3, None, 5, 4],
        [None,  None, 6, 4, None, 2, 7, None, None],
        [9, 7, None, 8, 5, None, None, 1, None],
        [None, 1, None, None, 8, 7, None, None, 9],
        [3, 4, None, 6, None, None, 8, None, None],
        [8, None, None, None, None, None, None, None, 1],
    ]


# Temporary - for testing
def show_output(board, file=None):
    n = len(board)
    print('-' * (4 * n))
    for i in range(0, n):
        for j in range(0, n):
            print('| {} '.format(board[i][j]), end='')
        print('|')
        print('-' * (4 * n))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initial_board = get_input(None)
    sudoku_solver = Sudoku(initial_board)
    output_board = sudoku_solver.solve()
    # show_output(sudoku_solver.solution)
    show_output(output_board)
    # show_output(sudoku_solver.board)
