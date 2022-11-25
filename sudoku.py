import datetime


class Sudoku:
    def __init__(self, board):
        """
        Initializes solver for n x n Sudoku puzzle

        :param
         board: matrix of integers of size n x n. Cells that does not present in initial board, must be set to None

        Constructs domain for each cell.
        """

        self.runtime = None
        self.n = len(board)
        self.board = [[]] * self.n
        for i in range(0, self.n):
            self.board[i] = [[]] * self.n
            for j in range(0, self.n):
                x = board[i][j]
                self.board[i][j] = [x] if x else list(range(1, self.n + 1))

        self._init_solution_board()

    def _init_solution_board(self):
        self.solution = [[None] * self.n] * self.n

    def is_valid(self):
        """Checks if input board is valid and solvable."""
        # TODO
        return True

    def is_valid_solution(self):
        """Checks if solution found by the algorithm is valid. (May be used for testing.)"""

    def _construct_algorithm(self, only_backtracking, backtracking_with_heuristic):
        """ Construct sequence of optimization algorithm is going to perform based on input parameters"""
        # TODO
        sequence = [self._ac3, self._naked_single, self._full_house, self._hidden_single, self._hidden_pair,
                    self._hidden_triples, self._locked_candidates, self._x_wing, self._backtracking]

        return sequence

    def solve(self, only_backtracking=False, backtracking_with_heuristic=True):
        start_time = datetime.datetime.now()
        self._init_solution_board()
        if not self.is_valid():
            print("Not valid Sudoku")
            return None

        sequence = self._construct_algorithm(only_backtracking, backtracking_with_heuristic)

        for action in sequence:
            result = action()
            if result:
                end_time = datetime.datetime.now()
                self.runtime = (end_time - start_time).total_seconds()
                return self.solution
            if result is False:
                end_time = datetime.datetime.now()
                self.runtime = (end_time - start_time).total_seconds()
                return None

        end_time = datetime.datetime.now()
        self.runtime = (end_time - start_time).total_seconds()
        return None

    def _ac3(self):
        """
        Executes AC3 algorithm.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing AC3')
        pass

    def _naked_single(self):
        """
        Executes Naked Single optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Naked Single')
        pass

    def _full_house(self):
        """
        Executes Full House optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Full House')
        pass

    def _hidden_single(self):
        """
        Executes Hidden Single optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Hidden Single')
        pass

    def _hidden_pair(self):
        """
        Executes Hidden Pair optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Hidden Pair')
        pass

    def _hidden_triples(self):
        """
        Executes Hidden Triples optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Hidden Triples')
        pass

    def _locked_candidates(self):
        """
        Executes Locked Candidates optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Locked Candidates')
        pass

    def _x_wing(self):
        """
        Executes X Wings optimization.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing X Wings')
        pass

    def _backtracking(self, with_heuristic=True):
        """
        Executes Backtracking.
        :return
        True - if after performing the algorithm puzzle is solved.
        False - if after performing the algorithm domain of some variable became empty.
        None - otherwise
        """
        print('executing Backtracking')
        pass
