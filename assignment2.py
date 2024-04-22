from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Callable, TypeAlias, Iterator
from functools import cached_property, lru_cache
from itertools import permutations, cycle
import heapq
import random
import copy

import numpy as np


Route: TypeAlias = Tuple[int, ...]
Cost: TypeAlias = float
Result: TypeAlias = Tuple[Cost, Route]


class Solver(ABC):

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
        for i in range(m-1, 1, -1):
            if (
                route[i-1] % 2 == 0 and route[i] % 2 == 1 and route[i-1] < self.n // 2
            ) or (
                route[i-1] % 2 == 1 and route[i] % 2 == 0 and route[i-1] >= self.n // 2
            ): 
                return False
        return True
    
    def get_all_feasible_routes(self) -> Iterator[Route]:   # expensive
        all_routes: Iterator[Route] = map(
            lambda x: (0,) + x + (self.n + 1,), 
            permutations(iterable=range(1, self.n + 1), r=self.n)
        )
        for route in all_routes:
            if self.is_feasible_route(route):
                yield route
    
    @abstractmethod
    def find_route(self) -> Result:
        pass


class BruteForce(Solver):

    # implement
    def find_route(self) -> Result:             # expensive
        results: List[Result] = []
        for route in self.get_all_feasible_routes():
            print(f'Evaluating: {route}')
            results.append((self.compute_cost(route), route))
        return min(results, key=lambda x: x[0])


class AStar(Solver):

    # implement
    def find_route(
        self,
        n_random_routes: int = 100,
        h_coeff: float = 0.8,
    ) -> Result:

        random_routes: Set[Route] = set()
        random_ints: List[int] = list(range(1, self.n + 1))
        while len(random_routes) < n_random_routes:
            random.shuffle(random_ints)
            random_route: Route = tuple([0] + random_ints + [self.n + 1])
            if self.is_feasible_route(random_route):
                random_routes.add(random_route)
            
        random_results: List[Result] = [(self.compute_cost(route), route) for route in set(random_routes)]
        best_cost, best_route = min(random_results, key=lambda x: x[0])

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
                    new_cost: Cost = self.compute_cost(new_route)
                    completed_edges: int = len(new_route) - 1
                    remaining_edges: int = self.n - 1 - completed_edges
                    heuristic: Cost = new_cost * remaining_edges / completed_edges * h_coeff
                    if new_cost + heuristic < best_cost:
                        heapq.heappush(pq, (new_cost, new_route))
                        print(f'Expand new_route: {new_route}')
                    else:
                        print(f'Prune from: {new_route}')
            
            print('End of expansion')
            print('----------------')

        return best_cost, best_route





if __name__ == '__main__':

    import time
    
    t0 = time.time()
    solver: Solver = AStar(csv_path='assignments/Assessment_II/data/I30.csv')
    r = solver.find_route_Astar(n_random_routes=100, h_coeff=0.8)
    print(r)
    print(f'Found solution in: {time.time() - t0} seconds')





