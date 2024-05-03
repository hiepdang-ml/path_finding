from typing import List, Set
import argparse
import heapq

from utils.functional import track_time
from utils.type_alias import Path, Result, Cost
from base import Solver


class CustomAStar(Solver):

    """
    Brute force solver for finding the optimal path.
    """

    def __init__(self, csv_path: str, n_random_paths: int, h_coeff: float) -> None:
        super().__init__(csv_path)
        self.n_random_paths: int = n_random_paths
        self.h_coeff: float = h_coeff

    @track_time
    def find_path(self) -> Result:
        if self.n_random_paths > 0:
            random_paths: Set[Path] = self.get_random_feasible_paths(n=self.n_random_paths)
            random_results: List[Result] = [(self.compute_cost(path), path) for path in random_paths]
            best_cost, _ = min(random_results, key=lambda x: x[0])
            # Manual garbage collect (to save some memory)
            del random_paths, random_results
        else:
            best_cost = float('inf')
                    
        # Initiate at node 0
        pq = [(0., [0])]

        while pq:
            # Evaluate best path so far
            cost, path = heapq.heappop(pq)
            print(f'Evaluating path: {path}')

            # If a complete path
            if len(path) == self.n + 2:
                print('A complete path')
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    
                print(f'Best (complete) path found so far: {best_path}, best_cost: {best_cost}')
                continue

            # If not a complete path, explore neighboring nodes
            for next_node in self.find_allowed_nodes(traveled_path=tuple(path)):
                new_path: List[int] = path + [next_node]
                print(f'Explore new_path: {new_path}')
                # Compute actual cost
                new_cost: Cost = self.compute_cost(path=tuple(new_path))
                # Compute heuristic cost
                completed_edges: int = len(new_path) - 1
                remaining_edges: int = self.n - 1 - completed_edges
                heuristic: Cost = new_cost * remaining_edges / completed_edges * self.h_coeff
                # Decide expand or prune
                if new_cost + heuristic < best_cost:
                    heapq.heappush(pq, (new_cost, new_path))
                    print(f'Expand new_path: {new_path}')
                else:
                    print(f'Prune from: {new_path}')
            
            print('End of expansion')
            print('----------------')

        return best_cost, best_path
    
    
def main() -> None:
    parser = argparse.ArgumentParser(description='Run Custom A-Star Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_random_paths', '-n', type=int, default=100, help='Number of random paths to initialize')
    parser.add_argument('--h_coeff', '-c', type=float, default=0.8, help='Heuristic coefficient')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = CustomAStar(**vars(args))
    r: Result = solver.find_path()
    print(f'Found solution: {r}')
    print(f'Took {solver.duration} seconds')


if __name__ == '__main__':
    main()
    

