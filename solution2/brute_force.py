from typing import List, Iterator

from .utils.type_alias import track_time, Result, Route
from .base import Solver

class BruteForce(Solver):

    @track_time
    def find_route(self) -> Result:     # expensive
        feasible_routes: Iterator[Route] = self.generate_all_feasible_routes()
        results: List[Result] = []
        for route in feasible_routes:
            print(f'Evaluating: {route}')
            results.append((self.compute_cost(route), route))
        return min(results, key=lambda x: x[0])


