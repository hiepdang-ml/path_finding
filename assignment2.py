from typing import List, Tuple, Callable, TypeAlias, Iterator
from functools import cached_property, lru_cache
from itertools import permutations
import heapq

import numpy as np


Route: TypeAlias = Tuple[int, ...]
Cost: TypeAlias = float
ResultSet: TypeAlias = Tuple[Cost, Route]


class Solver:

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        
        with open(file=csv_path, mode='r') as file:
            next(file)      # skip first row
            data: List[str] = file.readlines()
            pre_process = lambda s: [
                float(x) for x in s.replace('\n','').split(',')[-2:]
            ]
            data: List[List[float]] = list(map(pre_process, data))
            self.nodes: np.ndarray = np.array(data, dtype=np.float32)

        self.n: int = self.nodes.shape[0] - 2

    @cached_property
    def distances(self) -> np.ndarray:  # shape (n+2, n+2)
        diff: np.ndarray = (
            self.nodes[:, np.newaxis, :]    # shape (n+2, 1, 2)
            - self.nodes[np.newaxis, :, :]  # shape (1, n+2, 2)
        )
        return np.sqrt(np.sum(diff**2, axis=-1)).round(decimals=1)

    @lru_cache(maxsize=128, typed=True)
    def compute_D(self, route: Route) -> float:
        return self.distances[route[:-1], route[1:]].sum()

    @lru_cache(maxsize=128, typed=True)
    def compute_delta(self, route: Route) -> float:
        distances: np.ndarray = self.distances[route[:-1], route[1:]]
        return distances.max() - distances.min()
    
    @lru_cache(maxsize=128, typed=True)
    def compute_cost(self, route: Route, log: bool = False) -> Cost:
        delta: float = self.compute_delta(route)
        D: float = self.compute_D(route)
        cost: float = self.n * self.distances.max() * delta + D

        if log:
            print(f"Path : {'-'.join(map(str, route))}")
            print(f'Total distances : {D}')
        
        return cost
    
    def is_feasible_route(self, route: Route) -> bool:
        if route[0] != 0:
            return False
        if len(route) == self.n + 2 and route[-1] != self.n + 1:    # complete route
            return False
        if len(route) == self.n + 2:
            m: int = self.n + 1
        else:
            m: int = len(route)
        for i in range(2, m):
            if (
                route[i-1] % 2 == 0 and route[i] % 2 == 1 and route[i-1] < self.n // 2
            ) or (
                route[i-1] % 2 == 1 and route[i] % 2 == 0 and route[i-1] >= self.n // 2
            ): 
                return False
        return True
    
    @cached_property
    def feasible_routes(self) -> Iterator[Route]:
        all_routes: Iterator[Route] = map(
            lambda x: (0,) + x + (self.n + 1,), 
            permutations(iterable=range(1, self.n + 1), r=self.n)
        )
        return filter(self.is_feasible_route, all_routes)
    
    def find_route_brute_force(self) -> ResultSet:
        results: List[ResultSet] = []
        for route in self.feasible_routes:
            results.append((self.compute_cost(route), route))
        return min(results, key=lambda x: x[0])

    def find_route_dijkstra(self) -> ResultSet:
        # Initiate at node 0
        pq: List[ResultSet] = [(0., (0,))]
        best_cost = float('inf')
        best_route: Route | None = None
        
        while pq:
            # Evaluate best route so far
            cost, route = heapq.heappop(pq)
            print(f'Best route so far: {route}')

            # If a complete route
            if len(route) == self.n + 2:
                print('A complete route')
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                    print(f'Best (complete) route found: {best_route}, best_cost: {best_cost}')
                continue

            # If not a complete route, explore neighboring nodes
            for next_node in range(1, self.n + 2):
                if next_node in route:
                    continue

                new_route: Route = route + (next_node,)
                if self.is_feasible_route(new_route):
                    print(f'Expand to new_route: {new_route}')
                    new_cost: Cost = self.compute_cost(new_route)
                    heapq.heappush(pq, (new_cost, new_route))
            
            print('End of expansion')
            print('----------------')

        return best_cost, best_route




if __name__ == '__main__':
    self = Solver('assignments/Assessment_II/data/I20.csv')
    # r_f = self.find_route_brute_force()
    r_d = self.find_route_dijkstra()

