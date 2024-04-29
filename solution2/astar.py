from typing import List, Set
import argparse
import heapq

from utils.functional import track_time
from utils.type_alias import Route, Result, Cost
from base import Solver


class CustomAStar(Solver):

    def __init__(self, csv_path: str, n_random_routes: int, h_coeff: float) -> None:

        super().__init__(csv_path)
        self.n_random_routes: int = n_random_routes
        self.h_coeff: float =h_coeff

    @track_time
    def find_route(self) -> Result:
        random_routes: Set[Route] = self.get_random_feasible_routes(n=self.n_random_routes)
        random_results: List[Result] = [(self.compute_cost(route), route) for route in random_routes]
        best_cost, best_route = min(random_results, key=lambda x: x[0])

        # Manual garbage collect (to save some memory)
        del random_routes, random_results

        # Initiate at node 0
        pq: List[Result] = [(0., (0,))]

        while pq:
            print(f'Best (complete) route found so far: {best_route}, best_cost: {best_cost}')
            # Evaluate best route so far
            cost, route = heapq.heappop(pq)
            print(f'Evaluating route: {route}')

            # If a complete route
            if len(route) == self.n + 2:
                print('A complete route')
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                continue

            # If not a complete route, explore neighboring nodes
            for next_node in range(1, self.n + 2):
                if next_node in route:
                    continue

                new_route: Route = route + (next_node,)
                if self.is_feasible_route(new_route):
                    print(f'Explore new_route: {new_route}')
                    # Compute actual cost
                    new_cost: Cost = self.compute_cost(new_route)
                    # Compute heuristic cost
                    completed_edges: int = len(new_route) - 1
                    remaining_edges: int = self.n - 1 - completed_edges
                    heuristic: Cost = new_cost * remaining_edges / completed_edges * self.h_coeff
                    # Decide expand or prune
                    if new_cost + heuristic < best_cost:
                        heapq.heappush(pq, (new_cost, new_route))
                        print(f'Expand new_route: {new_route}')
                    else:
                        print(f'Prune from: {new_route}')
            
            print('End of expansion')
            print('----------------')

        return best_cost, best_route
    
    
def main() -> None:
    parser = argparse.ArgumentParser(description='Run Custom A-Star Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_random_routes', '-n', type=int, default=100, help='Number of random routes to initialize')
    parser.add_argument('--h_coeff', '-c', type=float, default=0.8, help='Heuristic coefficient')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = CustomAStar(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    main()
    

