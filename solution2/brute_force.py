from typing import List, Iterator
import argparse

from utils.type_alias import Result, Path, Cost
from utils.functional import track_time
from base import Solver

class BruteForce(Solver):

    @track_time
    def find_path(self) -> Result:     # expensive
        feasible_paths: Iterator[Path] = self.generate_all_feasible_paths()
        best_cost = float('inf')
        for path in feasible_paths:
            print(f'Evaluating: {path}')
            cost: Cost = self.compute_cost(path)
            if cost < best_cost:
                best_cost: Cost = cost
                best_path: Path = path
        return best_cost, best_path


def main():
    parser = argparse.ArgumentParser(description='Run Brute Force Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = BruteForce(**vars(args))
    r: Result = solver.find_path()
    solver.to_file(path= r[1])
    print(f'Found solution: {r}')
    print(f'Took {solver.duration} seconds')


if __name__ == '__main__':
    main()
