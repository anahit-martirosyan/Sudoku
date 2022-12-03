import datetime
import math
from enum import Enum


class VariableOrdering(Enum):
    NEXT_UNASSIGNED_CELL = 1
    MRV = 2
    DEGREE_HEURISTIC = 3

class ValueOrdering(Enum):
    NEXT_VALUE = 1
    LCV = 2

class Inference(Enum):
    NO_INFERENCE = 1
    FORWARD_CHECKING = 2
    MAC = 3


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

        self.cell_assignment_stack = []
        self.value_assignment_dict = {}
        self.backtracking_stack = []

        self.get_next_cell = None
        self.get_next_value = None
        self.inference_function = None


    def _init_solution_board(self):
        self.solution = []
        for i in range(0, self.n):
            self.solution.append([None] * self.n)
            for j in range(0, self.n):
                if self.initial_board[i][j]:
                    self.solution[i][j] = self.initial_board[i][j]

    def is_valid(self):
        """Checks if input board is valid and solvable."""
        # for 9x9, check if at least 17 clues are given
        if self.n == 9:
            ls = [i for j in self.initial_board for i in j]
            num_clues = sum(x is not None for x in ls)
            if num_clues < 17:
                return False

        # check rows
        for i in range(self.n):
            vals = set()
            for j in range(self.n):
                if self.initial_board[i][j] in vals:
                    return False
                if self.initial_board[i][j] is not None:
                    vals.add(self.initial_board[i][j])

        # check columns
        for j in range(self.n):
            vals = set()
            for i in range(self.n):
                if self.initial_board[i][j] in vals:
                    return False
                if self.initial_board[i][j] is not None:
                    vals.add(self.initial_board[i][j])

        # check sub-grids
        grid_n = int(math.sqrt(self.n))
        for i in range(grid_n):
            for j in range(grid_n):
                vals = set()
                start_x = i * grid_n
                start_y = j * grid_n

                for k in range(grid_n):
                    for m in range(grid_n):
                        val = self.initial_board[start_x + k][start_y + m]
                        if val in vals:
                            return False
                        if val is not None:
                            vals.add(val)
        return True

    def is_valid_solution(self):
        """Checks if solution found by the algorithm is valid."""
        # check if the corresponding values of the initial board and the solution are equal
        for i in range(self.n):
            for j in range(self.n):
                if self.initial_board[i][j] and self.solution[i][j]:
                    if self.solution[i][j] != self.initial_board[i][j]:
                        return False

        # check rows
        for i in range(self.n):
            vals = set()
            for j in range(self.n):
                if self.solution[i][j] in vals:
                    return False
                if self.solution[i][j] is not None:
                    vals.add(self.solution[i][j])

        # check columns
        for j in range(self.n):
            vals = set()
            for i in range(self.n):
                if self.solution[i][j] in vals:
                    return False
                if self.solution[i][j] is not None:
                    vals.add(self.solution[i][j])

        # check sub-grids
        grid_n = int(math.sqrt(self.n))
        for i in range(grid_n):
            for j in range(grid_n):
                vals = set()
                start_x = i * grid_n
                start_y = j * grid_n

                for k in range(grid_n):
                    for m in range(grid_n):
                        val = self.solution[start_x + k][start_y + m]
                        if val in vals:
                            return False
                        if val is not None:
                            vals.add(val)
        return True

    @staticmethod
    def show(board):
        if not board:
            print('No board')
        n = len(board)
        print('-' * (4 * n))
        for i in range(0, n):
            for j in range(0, n):
                print('| {} '.format(board[i][j]), end='')
            print('|')
            print('-' * (4 * n))

    def _construct_algorithm(self, backtracking_only, no_constraint_propagation):
        """ Construct sequence of optimizations which algorithm is going to perform based on input parameters"""
        if backtracking_only:
            sequence = [self._backtracking]
        elif no_constraint_propagation:
            sequence = [
                self._ac3,
                self._backtracking
            ]
        else:
            sequence = [
                self._ac3,
                self._naked_single,
                self._naked_pair,
                self._full_house,
                self._hidden_single,
                self._hidden_pair,
                self._hidden_triples,
                self._locked_candidates,
                self._x_wing,
                self._backtracking
            ]

        return sequence

    def _get_variable_ordering_function(self, variable_ordering):
        if variable_ordering == VariableOrdering.NEXT_UNASSIGNED_CELL:
            return self._get_next_unassigned_cell
        if variable_ordering == VariableOrdering.MRV:
            return self._get_next_cell_with_mrv
        # variable_ordering == VariableOrdering.DEGREE_HEURISTIC
        return self._get_next_cell_with_degree_heuristic

    def _get_value_ordering_function(self, value_ordering):
        return self._get_next_value_with_lcv if value_ordering == ValueOrdering.LCV \
            else self._get_next_value

    def _get_inference_function(self, inference):
        if inference == Inference.FORWARD_CHECKING:
            return self._forward_checking
        if inference == Inference.MAC:
            return self._maintaining_arc_consistency
        # inference == Inference.NO_INFERENCE
        return None

    def solve(self, backtracking_only=False, no_constraint_propagation=False, variable_ordering=VariableOrdering.NEXT_UNASSIGNED_CELL, value_ordering=ValueOrdering.NEXT_VALUE, inference=Inference.NO_INFERENCE):
        start_time = datetime.datetime.now()
        self._init_solution_board()
        self.get_next_cell = self._get_variable_ordering_function(variable_ordering)
        self.get_next_value = self._get_value_ordering_function(value_ordering)
        self.inference_function = self._get_inference_function(inference)

        if not self.is_valid():
            print("Not valid Sudoku")
            end_time = datetime.datetime.now()
            self.runtime = (end_time - start_time).total_seconds()
            return None

        sequence = self._construct_algorithm(backtracking_only, no_constraint_propagation)

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

    def _revert(self, i, j, initial_domain):
        self.solution[i][j] = None
        self.board[i][j] = initial_domain


    def _get_neighbors(self, cell):
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
                neighbors = self._get_neighbors((i, j))
                q += [((i, j), (k, h)) for (k, h) in neighbors]
        return q

    def _ac3_revise(self, first, second):
        first_domain = self.board[first[0]][first[1]]
        second_domain = self.board[second[0]][second[1]]
        old_domain = self.board[first[0]][first[1]].copy()
        revised = False

        if len(second_domain) == 1:
            # Second cell has fixed value, remove that value from first's domain
            if second_domain[0] in first_domain:
                first_domain.remove(second_domain[0])
                revised = True
        return revised, old_domain

    def _ac3(self, initial_constraints=None):
        """
        Executes AC3 algorithm.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise.
        """
        # print('executing AC3')
        old_domains = {}
        old_board = self.board.copy()
        q = initial_constraints or self._get_constraint_queue()
        while q:
            (first, second) = q.pop(0)
            revised, _ = self._ac3_revise(first, second)
            if revised:
                old_domains[first] = old_board[first[0]][first[1]]
                if not self.board[first[0]][first[1]]:
                    return False, old_domains
                neighbors = self._get_neighbors(first)
                q += [((k, h), first) for (k, h) in neighbors]

        return True, old_domains


    def _naked_single(self):
        """
        Executes Naked Single optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing Naked Single')
        q = [((i, j), None) for i in range(0, self.n) for j in range(0, self.n) if len(self.board[i][j]) == 1]
        while q:
            ((i, j), v) = q.pop(0)
            if v:
                if v in self.board[i][j]:
                    self.board[i][j].remove(v)
            if len(self.board[i][j]) == 1:
                self._set_value(i, j, self.board[i][j][0])
                neighbors = self._get_neighbors((i, j))
                q += [(neighbor, self.solution[i][j]) for neighbor in neighbors if self.solution[neighbor[0]][neighbor[1]] is None]

        return True


    def _naked_pair(self):
        """
        Executes Naked Pair optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing Naked Pair')
        grid_n = int(math.sqrt(self.n))

        q = [((i, j), self.board[i][j]) for i in range(0, self.n) for j in range(0, self.n) if len(self.board[i][j]) == 2]
        while q:
            ((i, j), values) = q.pop(0)
            neighbors = self._get_neighbors((i, j))
            same_domain_neighbors = [n for n in neighbors if self.board[n[0]][n[1]] == self.board[i][j]]
            for neighbor in same_domain_neighbors:
                same_constraint_group_neighbors = []
                if neighbor[0] == i:
                    # Same row
                    same_constraint_group_neighbors += [(i, h) for h in range(0, self.n)]
                elif neighbor[1] == j:
                    # Same column
                    same_constraint_group_neighbors += [(k, j) for k in range(0, self.n)
                                                        if (k, j) not in same_constraint_group_neighbors]
                if (i // grid_n) == (neighbor[0] // grid_n) and (j // grid_n) == (neighbor[1] // grid_n):
                    # Same grid
                    grid_x_index = i // grid_n
                    grid_y_index = j // grid_n
                    grid_start_x = grid_x_index * grid_n
                    grid_start_y = grid_y_index * grid_n
                    same_constraint_group_neighbors += [(k, h) for k in range(grid_start_x, grid_start_x + grid_n)
                                                            for h in range(grid_start_y, grid_start_y + grid_n)
                                                        if (k, h) not in same_constraint_group_neighbors]

                for k, h in same_constraint_group_neighbors:
                        if (k, h) != (i, j) and (k, h) != neighbor:
                            new_domain = [v for v in self.board[k][h] if v not in values]
                            if new_domain != self.board[k][h] and len(self.board[k][h]) == 2:
                                self.board[k][h] = new_domain
                                if (k, h) not in q:
                                    q.append(((k, h), self.board[k][h]))

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
        # print('executing Full House')
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

        return True

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
        # print('executing Hidden Single')
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
        # print('executing Hidden Pair')
        all_pairs = self._get_all_value_pairs()
        return self._hidden_single_pair_triple(all_pairs)

    def _hidden_triples(self):
        """
        Executes Hidden Triples optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing Hidden Triples')
        all_triples = self._get_all_value_triples()
        return self._hidden_single_pair_triple(all_triples)


    def _locked_candidates(self):
        """
        Executes Locked Candidates optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing Locked Candidates')
        grid_n = int(math.sqrt(self.n))
        # queue if grid indices
        q = [(i, j) for i in range(0, grid_n) for j in range(0, grid_n)]
        while q:
            (i, j) = q.pop(0)
            grid_start_x = i * grid_n
            grid_start_y = j * grid_n
            for v in range(1, self.n + 1):
                candidates = [(i, j) for i in range(grid_start_x, grid_start_x + grid_n)
                              for j in range(grid_start_y, grid_start_y + grid_n) if v in self.board[i][j]]

                if len(candidates) < grid_n:
                    q_set = set()
                    cells_to_alter = []
                    if len(set([i for i, _ in candidates])) == 1:
                        # all candidates are in the same row
                        k = candidates[0][0]
                        cells_to_alter += [(k, h) for h in list(range(0, grid_start_y)) +  list(range(grid_start_y + grid_n, self.n))]

                    elif len(set([j for _, j in candidates])) == 1:
                        # all candidates are in the same column
                        h = candidates[0][1]
                        cells_to_alter += [(k, h) for k in list(range(0, grid_start_x)) + list(range(grid_start_x + grid_n, self.n))]

                    for k, h in cells_to_alter:
                        if v in self.board[k][h]:
                            self.board[k][h].remove(v)
                            q_set.add((k // grid_n, h // grid_n))

                    q.extend(list(q_set))

    def _x_wing(self):
        """
        Executes X Wings optimization.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing X Wings')
        pass


    def _revert_inference(self):
        old_domains = self.backtracking_stack.pop()
        for (i, j), domain in old_domains.items():
            if not domain:
                print((i, j))
            self.board[i][j] = domain
            self.solution[i][j] = None

    def _forward_checking(self, cell):
        neighbors = self._get_neighbors(cell)
        old_domains = {}
        for n in neighbors:
            revised, old_domain = self._ac3_revise(n, cell)
            if revised:
                old_domains[n] = old_domain

        self.backtracking_stack.append(old_domains)


    def _maintaining_arc_consistency(self, cell):
        neighbors = self._get_neighbors(cell)
        initial_constraints = [(n, cell) for n in neighbors]
        revised, old_domains = self._ac3(initial_constraints)
        if revised:
            self.backtracking_stack.append(old_domains)


    def _get_next_cell(self, cell):
        if cell is None:
            return 0, 0

        i, j = cell
        i_next, j_next = i, j

        if i + 1 < self.n:
            i_next = i + 1
        else:
            i_next = 0
            if j + 1 < self.n:
                j_next = j + 1
            else:
                return None
        return i_next, j_next

    def _get_next_unassigned_cell(self):
        next_cell = self.cell_assignment_stack[-1] if self.cell_assignment_stack else None

        while True:
            next_cell = self._get_next_cell(next_cell)
            if next_cell is None:
                return None

            if self.solution[next_cell[0]][next_cell[1]] is None:
                self.cell_assignment_stack.append(next_cell)
                return next_cell

    def _get_next_cell_with_mrv(self):
        min_domain_size = self.n + 1
        min_i, min_j = None, None
        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.solution[i][j] is None:
                    domain_size = len(self.board[i][j])
                    if domain_size < min_domain_size:
                        min_domain_size = domain_size
                        min_i, min_j = i, j

        self.cell_assignment_stack.append((min_i, min_j))
        return min_i, min_j

    def _get_next_cell_with_degree_heuristic(self):
        max_degree = 0
        max_i, max_j = None, None
        for i in range(0, self.n):
            for j in range(0, self.n):
                if self.solution[i][j] is None:
                    degree = len([neighbor for neighbor in self._get_neighbors((i, j))
                                  if self.solution[neighbor[0]][neighbor[1]] is None])
                    if degree > max_degree:
                        max_degree = degree
                        max_i, max_j = i, j

        self.cell_assignment_stack.append((max_i, max_j))
        return max_i, max_j

    def _get_next_value(self, cell):
        self.value_assignment_dict[cell] = self.value_assignment_dict.get(cell, [0])
        last_value = self.value_assignment_dict[cell][-1]
        potential_next = last_value + 1
        while potential_next <= self.n:
            if potential_next in self.board[cell[0]][cell[1]]:
                self.value_assignment_dict[cell].append(potential_next)
                return potential_next
            potential_next += 1

        return None

    def _get_next_value_with_lcv(self, cell):
        self.value_assignment_dict[cell] = self.value_assignment_dict.get(cell, [0])
        considered_values = self.value_assignment_dict[cell]
        neighbours = self._get_neighbors(cell)

        min_num = self.n
        result_value = None

        for v in self.board[cell[0]][cell[1]]:
            if v not in considered_values:
                num_of_changes = len([n for n in neighbours if v in self.board[n[0]][n[1]]])
                if num_of_changes < min_num:
                    min_num = num_of_changes
                    result_value = v

        return result_value

    def _backtracking(self):
        """
        Executes Backtracking.
        :return
        False - if after performing the algorithm domain of some variable became empty.
        True - otherwise
        """
        # print('executing Backtracking')
        if self._is_solved():
            return True

        next_cell = self.get_next_cell()

        # print('next_cell: {}'.format(next_cell))

        # print('variable_ordering {}'.format(self.variable_ordering))

        if next_cell is None:
            return True
        (i, j) = next_cell

        v = self.get_next_value(next_cell)

        initial_domain = self.board[i][j]

        while v:
            domain = self.board[i][j].copy()
            self._set_value(i, j, v)
            if self.is_valid_solution():
                if self.inference_function:
                    self.inference_function((i, j))
                if self.is_valid_solution():
                    res = self._backtracking()
                    if res:
                        return True

                domain.remove(v)
                self._revert(i, j, domain)

            v = self.get_next_value((i, j))

        self._revert(i, j, initial_domain)
        self.value_assignment_dict.pop(next_cell)
        self.cell_assignment_stack.pop()
        return False
