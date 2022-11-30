import datetime
import math


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
        self.initial_board = board
        self.board = [[]] * self.n
        for i in range(0, self.n):
            self.board[i] = [[]] * self.n
            for j in range(0, self.n):
                x = board[i][j]
                self.board[i][j] = [x] if x else list(range(1, self.n + 1))

        self._init_solution_board()

    def _init_solution_board(self):
        self.solution = []
        for i in range(0, self.n):
            self.solution.append([None] * self.n)
            for j in range(0, self.n):
                if self.initial_board[i][j]:
                    self.solution[i][j] = self.initial_board[i][j]

    def is_valid(self):
        """Checks if input board is valid and solvable."""
        # TODO
        return True

    def is_valid_solution(self):
        """Checks if solution found by the algorithm is valid. (May be used for testing.)"""

    def _construct_algorithm(self, only_backtracking, backtracking_with_heuristic):
        """ Construct sequence of optimization algorithm is going to perform based on input parameters"""
        # TODO
        sequence = [
            self._ac3,
            self._naked_single,
            self._full_house,
            self._hidden_single,
            self._hidden_pair,
            self._hidden_triples,
            self._locked_candidates,
            self._x_wing,
            self._backtracking]

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
            if result and self._is_solved():
                end_time = datetime.datetime.now()
                self.runtime = (end_time - start_time).total_seconds()
                return self.solution
            if not result:
                end_time = datetime.datetime.now()
                self.runtime = (end_time - start_time).total_seconds()
                return None

        end_time = datetime.datetime.now()
        self.runtime = (end_time - start_time).total_seconds()
        return None

    def _is_solved(self):
        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.solution[i][j] is None:
                    return False
        return True

    def _set_value(self, i, j, v):
        self.solution[i][j] = v
        self.board[i][j] = [v]

    def _ac3_get_neighbors(self, cell):
        """
        :param cell: (i, j) cell of puzzle board
        :return: list of (k, h), where (k, h) is cell in the same row, column or sub-grid as (i, j)
        """
        i, j = cell
        neighbors = set()
        # add row constraints
        for k in range(0, self.n):
            if k != i:
                neighbors.add((k, j))
        # add column constraints
        for k in range(0, self.n):
            if k != j:
                neighbors.add((i, k))
        # add sub-grid constraints
        grid_x_start = i - (i % int(math.sqrt(self.n)))
        grid_y_start = j - (j % int(math.sqrt(self.n)))
        for k in range(grid_x_start, grid_x_start + 3):
            for h in range(grid_y_start, grid_y_start + 3):
                if k != i or h != j:
                    neighbors.add((k, h))

        return list(neighbors)

    def _get_constraint_queue(self):
        q = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                neighbors = self._ac3_get_neighbors((i, j))
                q += [((i, j), (k, h)) for (k, h) in neighbors]
        return q

    def _ac3_revise(self, first, second):
        first_domain = self.board[first[0]][first[1]]
        second_domain = self.board[second[0]][second[1]]
        revised = False

        if len(second_domain) == 1:
            # Second cell has fixed value, remove that value from first's domain
            if second_domain[0] in first_domain:
                first_domain.remove(second_domain[0])
                revised = True
        return revised


    def _ac3(self):
        """
        Executes AC3 algorithm.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise.
        """
        print('executing AC3')
        q = self._get_constraint_queue()
        while q:
            (first, second) = q.pop(0)
            if self._ac3_revise(first, second):
                if not self.board[first[0]][first[1]]:
                    return False
                neighbors = self._ac3_get_neighbors(first)
                q += [((k, h), first) for (k, h) in neighbors]

        return True

    def _naked_single(self):
        """
        Executes Naked Single optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Naked Single')
        q = [((i, j), None) for i in range(0, self.n) for j in range(0, self.n) if len(self.board[i][j]) == 1]
        while q:
            ((i, j), v) = q.pop(0)
            if v:
                if v in self.board[i][j]:
                    self.board[i][j].remove(v)
            if len(self.board[i][j]) == 1:
                self._set_value(i, j, self.board[i][j][0])
                neighbors = self._ac3_get_neighbors((i, j))
                q += [(neighbor, self.solution[i][j]) for neighbor in neighbors if self.solution[neighbor[0]][neighbor[1]] is None]

        return True

    def _get_missing_value(self, data):
        for i in range(1, self.n + 1):
            if  i not in data:
                return i

        assert False

    def _full_house(self):
        """
        Executes Full House optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Full House')
        # Checking rows
        for i in range(0, self.n):
            row_values = self.solution[i]
            if row_values.count(None) == 1:
                v = self._get_missing_value(self.solution[i])
                unassigned_var_y = row_values.index(None)
                if v in self.board[i][unassigned_var_y]:
                    self._set_value(i, unassigned_var_y, v)
                else:
                    return False

        # Checking columns
        for j in range(0, self.n):
            col_values = [self.solution[i][j] for i in range(0, self.n)]
            if col_values.count(None) == 1:
                v = self._get_missing_value(col_values)
                unassigned_var_x = col_values.index(None)
                if v in self.board[unassigned_var_x][j]:
                    self._set_value(unassigned_var_x, j, v)
                else:
                    return False

        # Checking sub-grids
        grid_n = int(math.sqrt(self.n))
        for i in range(0, int(math.sqrt(self.n))):
            for j in range(0, int(math.sqrt(self.n))):
                grid_start_x = i * grid_n
                grid_start_y = j * grid_n

                grid_values = [self.solution[grid_start_x + k][grid_start_y + h]
                               for k in range(0, grid_n) for h in range(0, grid_n)]
                if grid_values.count(None) == 1:
                    v = self._get_missing_value(grid_values)
                    unassigned_var_x = grid_start_x + grid_values.index(None) % grid_n
                    unassigned_var_y = grid_start_y + grid_values.index(None) // grid_n
                    # for k in range(grid_start_x, grid_start_x + grid_n):
                    #     for h in range(grid_start_y, grid_start_y + grid_n):
                    #         if self.solution[k][h] is None:
                    #             unassigned_var_x = k
                    #             unassigned_var_y = h

                    if v in self.board[unassigned_var_x][unassigned_var_y]:
                        self._set_value(unassigned_var_x, unassigned_var_y, v)
                    else:
                        return False

    def _get_all_values(self):
        return [(i, ) for i in range(1, self.n + 1)]

    def _get_all_value_pairs(self):
         return [(i, j) for i in range(1, self.n) for j in range(i + 1, self.n + 1)]

    def _get_all_value_triples(self):
        return [(i, j, k) for i in range(1, self.n - 1) for j in range(i + 1, self.n) for k in range(j + 1, self.n + 1)]

    @staticmethod
    def _get_value_domain_indices(values, domains):
        """
        Checks if value v presents in the domain of only one variable
        :param values: tuple of values being checked
        :param domains: domains of variable that are in the same row, column or sub-grid
        :return: List of indices of the domain containing values
        """
        indices = set()
        for v in values:
            for i, domain in enumerate(domains):
                if v in domain:
                    indices.add(i)
        return list(indices)

    def _set_domains(self, values, cells):
        """
        Set domains of cells to be equal to values
        :param values: list of values to be set
        :param cells: indices indicating cells whose domains are going to be affected
        :return: 
        """
        for (i, j) in cells:
            self.board[i][j] = list(set(self.board[i][j]) & set(values))

    def _hidden_single_pair_triple(self, all_values):
        preferred_domain_size = len(all_values[0])
        # Checking rows
        for i in range(0, self.n):
            row_domains = self.board[i]
            for v in all_values:
                indices = self._get_value_domain_indices(v, row_domains)
                if not indices:
                    return False
                # print('value: {}, indices: {}'.format(v, indices))
                if len(indices) == preferred_domain_size:
                    self._set_domains(list(v), [(i, j) for j in indices])


        # Checking columns
        for j in range(0, self.n):
            col_domains = [self.board[i][j] for i in range(0, self.n)]
            for v in all_values:
                indices = self._get_value_domain_indices(v, col_domains)
                if not indices:
                    return False
                if len(indices) == preferred_domain_size:
                    self._set_domains(list(v), [(i, j) for i in indices])


        # Checking sub-grids
        grid_n = int(math.sqrt(self.n))
        for i in range(0, int(math.sqrt(self.n))):
            for j in range(0, int(math.sqrt(self.n))):
                grid_start_x = i * grid_n
                grid_start_y = j * grid_n

                grid_domains = [self.board[grid_start_x + k][grid_start_y + h]
                               for k in range(0, grid_n) for h in range(0, grid_n)]
                for v in all_values:
                    indices = self._get_value_domain_indices(v, grid_domains)
                    if not indices:
                        return False
                    if len(indices) == preferred_domain_size:
                        self._set_domains(list(v), [(grid_start_x + i % grid_n, grid_start_x + i % grid_n) for i in indices])


    def _hidden_single(self):
        """
        Executes Hidden Single optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Hidden Single')
        all_values = self._get_all_values()
        if self._hidden_single_pair_triple(all_values):
            return self._naked_single()
        return False

    def _hidden_pair(self):
        """
        Executes Hidden Pair optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Hidden Pair')
        all_pairs = self._get_all_value_pairs()
        return self._hidden_single_pair_triple(all_pairs)

    def _hidden_triples(self):
        """
        Executes Hidden Triples optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Hidden Triples')
        all_triples = self._get_all_value_triples()
        return self._hidden_single_pair_triple(all_triples)

    def _locked_candidates(self):
        """
        Executes Locked Candidates optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Locked Candidates')
        pass

    def _x_wing(self):
        """
        Executes X Wings optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing X Wings')
        pass

    def _backtracking(self, with_heuristic=True):
        """
        Executes Backtracking.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        print('executing Backtracking')
        pass
