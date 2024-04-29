from typing import List, Iterator
import argparse

from utils.type_alias import Result, Route
from utils.functional import track_time
from base import Solver

class BruteForce(Solver):

    @track_time
    def find_route(self) -> Result:     # expensive
        feasible_routes: Iterator[Route] = self.generate_all_feasible_routes()
        results: List[Result] = []
        for route in feasible_routes:
            print(f'Evaluating: {route}')
            results.append((self.compute_cost(route), route))
        return min(results, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description='Run Brute Force Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = BruteForce(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    main()
