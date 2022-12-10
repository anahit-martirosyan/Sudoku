from input_output import CSVInputProcessor2
from sudoku import Sudoku, VariableOrdering, ValueOrdering, Inference

RANKED_PUZZLES_DIR = 'Datasets/PuzzlesWithRanks/'
EASY_SUDOKU_1000 = 'sudoku_easy1000.csv'
MEDIUM_SUDOKU_1000 = 'sudoku_medium1000.csv'
HARD_SUDOKU = 'sudoku_hard.csv'

# For testing
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
    min_runtime = 10000.0
    max_runtime = 0
    for i in range(num_puzzles):
        board = input_processor.get_next_puzzle()
        sudoku = Sudoku(board)
        sudoku.solve(**kwargs)
        if not sudoku.is_solved() or not sudoku.is_valid_solution():
            print('Invalid solution')
            sudoku.show(sudoku.solution)
            sudoku.show(sudoku.board)

        runtime += sudoku.runtime
        if sudoku.runtime < min_runtime:
            min_runtime = sudoku.runtime
        if sudoku.runtime > max_runtime:
            max_runtime = sudoku.runtime
        print('puzzle #{}, runtime: {} s'.format(i, sudoku.runtime))

    return 'average runtime: {}s, min runtime: {}s, max runtime: {}'.format(runtime / num_puzzles, min_runtime, max_runtime)


def test_easy_puzzles():
    csv_processor = CSVInputProcessor2(RANKED_PUZZLES_DIR + EASY_SUDOKU_1000)
    num_puzzles = 100

    full_test(csv_processor, num_puzzles)

def test_medium_puzzles():
    csv_processor = CSVInputProcessor2(RANKED_PUZZLES_DIR + MEDIUM_SUDOKU_1000)
    num_puzzles = 100

    full_test(csv_processor, num_puzzles)

def test_hard_puzzles():
    csv_processor = CSVInputProcessor2(RANKED_PUZZLES_DIR + HARD_SUDOKU)
    num_puzzles = 10

    full_test(csv_processor, num_puzzles)


def full_test(processor, num_puzzles=None):

    avg_runtime = solve_sudoku(processor, num_puzzles)
    print('run: {} s'.format(avg_runtime))
    print()
    processor.restart_solver()

    ###
    avg_runtime = solve_sudoku(processor, num_puzzles, inference=Inference.FORWARD_CHECKING)
    print('run with forward checking: {} s'.format(avg_runtime))
    print()
    processor.restart_solver()

    ###
    avg_runtime = solve_sudoku(processor, num_puzzles, inference=Inference.MAC)
    print('run with MAC: {} s'.format(avg_runtime))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                variable_ordering=VariableOrdering.MRV)
    print('run with mrv: {} s'.format(avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking = solve_sudoku(processor, num_puzzles,
                                            value_ordering=ValueOrdering.LCV)
    print('run with lcv: {} s'.format(avg_runtime_backtracking))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                variable_ordering=VariableOrdering.MRV,
                                                value_ordering=ValueOrdering.LCV)
    print('run with mrv and lcv: {} s'.format(avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ##
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.FORWARD_CHECKING,
                                                variable_ordering=VariableOrdering.MRV)
    print('run with forward checking with mrv: {} s'.format(
        avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking = solve_sudoku(processor, num_puzzles,
                                            inference=Inference.FORWARD_CHECKING,
                                            value_ordering=ValueOrdering.LCV)
    print('run with forward checking with lcv: {} s'.format(
        avg_runtime_backtracking))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.FORWARD_CHECKING,
                                                variable_ordering=VariableOrdering.MRV,
                                                value_ordering=ValueOrdering.LCV)
    print('run with forward checking with mrv and lcv: {} s'.format(
        avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.MAC,
                                                variable_ordering=VariableOrdering.MRV)
    print('run with MAC with mrv: {} s'.format(avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ###
    avg_runtime_backtracking = solve_sudoku(processor, num_puzzles,
                                            inference=Inference.MAC,
                                            value_ordering=ValueOrdering.LCV)
    print('run with MAC with lcv: {} s'.format(avg_runtime_backtracking))
    print()
    processor.restart_solver()

    ##
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.MAC,
                                                variable_ordering=VariableOrdering.MRV,
                                                value_ordering=ValueOrdering.LCV)
    print('run with MAC with mrv and lcv: {} s'.format(
        avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()



    ###
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.MAC,
                                                variable_ordering=VariableOrdering.MRV,
                                                value_ordering=ValueOrdering.RANDOM_VALUE)
    print('run with MAC with mrv and random value ordering: {} s'.format(avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()

    ##
    avg_runtime_backtracking_mrv = solve_sudoku(processor, num_puzzles,
                                                inference=Inference.FORWARD_CHECKING,
                                                variable_ordering=VariableOrdering.MRV,
                                                value_ordering=ValueOrdering.RANDOM_VALUE)
    print('run with forward checking with mrv and random value ordering: {} s'.format(
        avg_runtime_backtracking_mrv))
    print()
    processor.restart_solver()


if __name__ == '__main__':
    test_easy_puzzles()
    test_medium_puzzles()
    test_hard_puzzles()

