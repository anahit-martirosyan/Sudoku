from input_output import CSVInputProcessor, TXTInputProcessor
from sudoku import Sudoku, VariableOrdering, ValueOrdering, Inference

TRIVIAL_PUZZLES_FILE = '../Datasets/TrivialPuzzles1mln/sudoku.csv'
HARD_PUZZLES_DIR = '../Datasets/HardPuzzles/'
HARD_PUZZLES = ['87hard18clue.txt', '95hard.txt', '234hard.txt', '1465hard.txt']
# HARD18CLUE = '18clue_77.txt'

# Temporary - for testing
def get_input():
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
        # print('puzzle #{}, runtime: {} s'.format(i, sudoku.runtime))

    return runtime / num_puzzles

def test_sudoku(processor, num_puzzles=None):
    avg_runtime = solve_sudoku(processor, num_puzzles,
                               no_constraint_propagation=True)
    print('average runtime: {} s'.format(avg_runtime))
    processor.restart_solver()

    avg_runtime = solve_sudoku(processor, num_puzzles, inference=Inference.FORWARD_CHECKING,
                               no_constraint_propagation=True)
    print('average runtime with forward checking: {} s'.format(avg_runtime))
    processor.restart_solver()

    avg_runtime = solve_sudoku(processor, num_puzzles, inference=Inference.MAC,
                               no_constraint_propagation=True)
    print('average runtime with MAC: {} s'.format(avg_runtime))
    processor.restart_solver()
    # avg_runtime_backtracking = solve_sudoku(processor, num_puzzles,  #backtracking_only=True
    #                                         )
    # print('average runtime - backtracking only: {} s'.format(avg_runtime_backtracking))
    #
    # processor.restart_solver()
    # avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,  #backtracking_only=True,
    #                                             variable_ordering=VariableOrdering.MRV)
    # print('average runtime - backtracking only with mrv: {} s'.format(avg_runtime_backtracking_mrv))
    #
    # # processor.restart_solver()
    # # avg_runtime_backtracking_dh = solve_sudoku(processor, num_puzzles,  backtracking_only=True,
    # #                                             variable_ordering=VariableOrdering.DEGREE_HEURISTIC)
    # # print('average runtime - backtracking only with degree heuristic: {} s'.format(avg_runtime_backtracking_dh))
    #
    # processor.restart_solver()
    # avg_runtime_backtracking = solve_sudoku(processor, num_puzzles,  #backtracking_only=True,
    #                                         value_ordering=ValueOrdering.LCV)
    # print('average runtime - backtracking only with lcv: {} s'.format(avg_runtime_backtracking))
    #
    # processor.restart_solver()
    # avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles, # backtracking_only=True,
    #                                             variable_ordering=VariableOrdering.MRV, value_ordering=ValueOrdering.LCV)
    # print('average runtime - backtracking only with mrv and lcv: {} s'.format(avg_runtime_backtracking_mrv))
    #
    # # processor.restart_solver()
    # # avg_runtime_backtracking_dh = solve_sudoku(processor, num_puzzles,  backtracking_only=True,
    # #                                             variable_ordering=VariableOrdering.DEGREE_HEURISTIC, value_ordering=ValueOrdering.LCV)
    # # print('average runtime - backtracking only with degree heuristic: {} s'.format(avg_runtime_backtracking_dh))

def test_trivial_puzzles():
    csv_processor = CSVInputProcessor(TRIVIAL_PUZZLES_FILE)
    num_puzzles = 100
    test_sudoku(csv_processor, num_puzzles)



def test_87hard18clue_puzzles():
    for filename in HARD_PUZZLES:
        print(filename)
        txt_processor = TXTInputProcessor(HARD_PUZZLES_DIR + filename)
        test_sudoku(txt_processor)


if __name__ == '__main__':

    # test_trivial_puzzles()
    test_87hard18clue_puzzles()
