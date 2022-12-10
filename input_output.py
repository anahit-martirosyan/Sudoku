import csv


class InputProcessor:
    """As we will use different datasets, we may process inputs of each dataset differently."""
    def __init__(self, puzzle_size = 9):
        self.input_puzzles = []
        self.current_puzzle_index = 0
        self.current_puzzle = None
        self.puzzle_size = puzzle_size

    def restart_solver(self):
        self.current_puzzle_index = 0
        self.current_puzzle = None

class CSVInputProcessor(InputProcessor):
    def __init__(self, filename, puzzle_size = 9):
        super(CSVInputProcessor, self).__init__(puzzle_size)

        self.empty_cell_value ='0'
        self.solutions = []
        self.current_solution = None

        self._read_file(filename)

    def _read_file(self, filename):
        with open(filename) as inputs_file:
            csv_reader = csv.reader(inputs_file)
            # skipping headlines
            csv_reader.__next__()
            for row in csv_reader:
                self.input_puzzles.append(row[0])
                self.solutions.append(row[1])

    def restart_solver(self):
        super(CSVInputProcessor, self).restart_solver()
        self.current_solution = None

    def get_next_puzzle(self):
        puzzle_str = self.input_puzzles[self.current_puzzle_index]
        solution_str = self.solutions[self.current_puzzle_index]
        self.current_puzzle = [[]] * self.puzzle_size
        self.current_solution = [[]] * self.puzzle_size

        index = 0
        for i in range(0, self.puzzle_size):
            self.current_puzzle[i] = [None] * self.puzzle_size
            self.current_solution[i] = [None] * self.puzzle_size

            for j in range(0, self.puzzle_size):
                if puzzle_str[index] != self.empty_cell_value:
                    self.current_puzzle[i][j] = int(puzzle_str[index])
                self.current_solution[i][j] = int(solution_str[index])
                index += 1

        self.current_puzzle_index += 1

        return self.current_puzzle


class CSVInputProcessor2(CSVInputProcessor):
    def __init__(self, filename, puzzle_size = 9):
        super(CSVInputProcessor2, self).__init__(filename, puzzle_size)

        self.ranks = []
        self.num_clues = []
        self.empty_cell_value ='.'


    def _read_file(self, filename):
        with open(filename) as inputs_file:
            csv_reader = csv.reader(inputs_file)
            # skipping headlines
            csv_reader.__next__()
            for row in csv_reader:
                row = row[0].split('\t')
                self.input_puzzles.append(row[2])
                self.solutions.append(row[3])
                # self.num_clues.append(row[4])
                # self.ranks.append(row[5])


class TXTInputProcessor(InputProcessor):
    def __init__(self, filename, puzzle_size = 9):
        super(TXTInputProcessor, self).__init__(puzzle_size)
        with open(filename) as inputs_file:
            for line in inputs_file:
                self.input_puzzles.append(line)


    def get_next_puzzle(self):
        if self.current_puzzle_index == len(self.input_puzzles):
            return None
        puzzle_str = self.input_puzzles[self.current_puzzle_index]
        self.current_puzzle = [[]] * self.puzzle_size

        index = 0
        for i in range(0, self.puzzle_size):
            self.current_puzzle[i] = [None] * self.puzzle_size

            for j in range(0, self.puzzle_size):
                if puzzle_str[index] != '.':
                    self.current_puzzle[i][j] = int(puzzle_str[index])
                index += 1

        self.current_puzzle_index += 1

        return self.current_puzzle
