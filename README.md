# SUDOKU Solver

Here is a solver written in python aimed to solve Sudoku puzzled.

## Algorithm and parameters

Sudoku puzzle is formulated as CSP problem, and the main algorithm for solving the puzzle is backtracking.
Before performing backtracking AC-3 algorithm is performed to gain arc consistency. After AC-3 the following constrained propagation rules are forced:
* Naked single
* Naked pair
* Full house
* Hidden single
* Hidden pair
* Hidden triple
* Locked candidates

Solver allows skipping constraint propagation phase by setting `no_constraint_propagation` to `True`,
and it is possible to skip also AC-3, by setting `backtracking_only` to `True`.

For the backtracking the following options are available:

Variable ordering
* `NEXT_UNASSIGNED_CELL` - when choosing a cell to assign a value, next cell in the order is chosen
* `MRV` - Minimum Remaining Values - when choosing a cell to assign a value, the variable with the fewest legal values is chosen

Value ordering
* `NEXT_VALUE` - when choosing a value to assign to a variable, the next value in increasing order is chosen
* `RANDOM_VALUE` - when choosing a value to assign to a variable, a random variable from the domain is chosen
* `LCV` - Least Constraining Values - when choosing a value to assign to a variable, the value with rules out the fewest choises for the neighboring cells is chosen

Inference algorithms
* `NO_INFERENCE` - no inference algorithm is performed
* `FORWARD_CHECKING` - after assigning a value to a variable arc consistency for that variable is established
* `MAC` - Maintaining Arc Consistency - after assigning a value to a variable arc consistency for the whole board is established

Default values for the mentioned parameters are:  
Variable ordering - `MRV`  
Value ordering - `LCV`  
Inference algorithm - `FORWARD_CHECKING`  

## Datasets

The solver is tested using Sudoku puzzles in `Datasets/PuzzlesWithRanks` directory. Puzzles are separated to 3 difficulty levels based on their ranks:
* Easy - ranks 0-4
* Medium - ranks 4-6
* Hard - ranks 6-10

Tests are written to compare the performance of the algorithm with all possible combinations of backtracking parameters on Easy (100 puzzles), Medium (100 puzzles) and Hard (10 puzzles) ranked Sudoku puzzles. 