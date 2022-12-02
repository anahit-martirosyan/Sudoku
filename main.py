from input_output import CSVInputProcessor, TXTInputProcessor
from sudoku import Sudoku

TRIVIAL_PUZZLES_FILE = '../Datasets/TrivialPuzzles1mln/sudoku.csv'
HARD_PUZZLES_DIR = '../Datasets/HardPuzzles/'
HARD18CLUE = '87hard18clue.txt'

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



def solve_sudoku(input_processor, num_puzzles=None, **kwargs):
    num_puzzles = num_puzzles or len(input_processor.input_puzzles)
    runtime = 0
    for i in range(num_puzzles):
        board = input_processor.get_next_puzzle()
        sudoku = Sudoku(board)
        sudoku.solve(**kwargs)
        if not sudoku.is_valid_solution():
            print('Invalid solution')

        runtime += sudoku.runtime
        print('puzzle #{}, runtime: {} s'.format(i, sudoku.runtime))

    return runtime / num_puzzles


def test_trivial_puzzles():
    csv_processor = CSVInputProcessor(TRIVIAL_PUZZLES_FILE)
    num_puzzles = 100
    avg_runtime = solve_sudoku(csv_processor, num_puzzles)
    print('average runtime: {} s'.format(avg_runtime))

    csv_processor.restart_solver()
    avg_runtime_backtracking = solve_sudoku(csv_processor, num_puzzles,  backtracking_only=True)
    print('average runtime - backtracking only: {} s'.format(avg_runtime_backtracking))


def test_87hard18clue_puzzles():
    txt_processor = TXTInputProcessor(HARD_PUZZLES_DIR + HARD18CLUE)
    avg_runtime = solve_sudoku(txt_processor)
    print('average runtime: {} s'.format(avg_runtime))

    txt_processor.restart_solver()
    avg_runtime_backtracking = solve_sudoku(txt_processor, backtracking_only=True)
    print('average runtime - backtracking only: {} s'.format(avg_runtime_backtracking))

if __name__ == '__main__':

    # test_trivial_puzzles()
    test_87hard18clue_puzzles()