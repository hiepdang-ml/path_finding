from typing import List, Iterator
import argparse

from utils.type_alias import Result, Path
from utils.functional import track_time
from base import Solver

class BruteForce(Solver):

    @track_time
    def find_path(self) -> Result:     # expensive
        feasible_paths: Iterator[Path] = self.generate_all_feasible_paths()
        results: List[Result] = []
        for path in feasible_paths:
            print(f'Evaluating: {path}')
            results.append((self.compute_cost(path), path))
        return min(results, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description='Run Brute Force Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = BruteForce(**vars(args))
    r: Result = solver.find_path()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    main()
