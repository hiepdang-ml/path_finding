from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Iterator, Any, Optional, Callable
from functools import cached_property
from itertools import permutations
import random

import numpy as np
from numpy.typing import NDArray

from utils.type_alias import Path, Cost, Result


class Solver(ABC):

    def __init__(self, csv_path: str) -> None:
        self.csv_path: str = csv_path
        
        with open(file=csv_path, mode='r') as file:
            next(file)      # skip first row
            data: List[str] = file.readlines()

        pre_process: Callable[[str], List[float]] = lambda s: [
            float(x) for x in s.replace('\n', '').split(',')[-2:]
        ]
        data: List[List[float]] = list(map(pre_process, data))
        self.nodes = np.array(data, dtype=np.float32)

        self.n: int = self.nodes.shape[0] - 2

    @cached_property
    def distances(self) -> NDArray[np.float32]:  # shape (n+2, n+2)
        diff: NDArray[np.float32] = (
            self.nodes[:, np.newaxis, :]    # shape (n+2, 1, 2)
            - self.nodes[np.newaxis, :, :]  # shape (1, n+2, 2)
        )
        return np.sqrt(np.sum(diff**2, axis=-1)).round(decimals=1)

    def compute_D(self, route: Path) -> float:
        return self.distances[route[:-1], route[1:]].sum()

    def compute_delta(self, route: Path) -> float:
        distances: NDArray[np.float32] = self.distances[route[:-1], route[1:]]
        return distances.max() - distances.min()
    
    def compute_cost(self, route: Path, log: bool = False) -> Cost:
        delta: float = self.compute_delta(route)
        D: float = self.compute_D(route)
        cost: float = self.n * self.distances.max() * delta + D
        if log:
            print(f"Path : {'-'.join(map(str, route))}")
            print(f'Total distances : {D}')
        return cost
    
    def is_feasible_route(self, route: Path) -> bool:
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
    
    def generate_all_feasible_routes(self) -> Iterator[Path]:   # expensive
        all_routes: Iterator[Path] = map(
            lambda x: (0,) + x + (self.n + 1,), 
            permutations(iterable=range(1, self.n + 1), r=self.n)
        )
        for route in all_routes:
            if self.is_feasible_route(route):
                yield route

    def get_random_feasible_routes(self, n: int, seed: Optional[int] = None) -> Set[Path]:
        random.seed(seed)
        random_routes: Set[Path] = set()
        random_ints: List[int] = list(range(1, self.n + 1))
        while len(random_routes) < n:
            random.shuffle(random_ints)
            random_route: Path = tuple([0] + random_ints + [self.n + 1])
            if self.is_feasible_route(random_route):
                random_routes.add(random_route)
        return random_routes
    
    def is_feasbile_transition(self, from_node: int, to_node: int) -> bool:
        if from_node == self.n + 1:
            return False
        if to_node == 0:
            return False
        if from_node == to_node:
            return False
        if (from_node == 0) != (to_node == self.n + 1): # logical XOR
            return True
        if (from_node == 0) and (to_node == self.n + 1):
            return False
        if from_node < self.n // 2 and from_node % 2 == 0 and to_node % 2 == 1:
            return False
        if from_node >= self.n // 2 and from_node % 2 == 1 and to_node % 2 == 0:
            return False
        return True
    
    @cached_property
    def node_groups(self) -> Tuple[Set[int]]:
        group0: Set[int] = {0, self.n + 1}
        group1: Set[int] = set()
        group2: Set[int] = set()
        group3: Set[int] = set()
        group4: Set[int] = set()

        for node in range(1, self.n + 1):
            if node < self.n // 2:
                if node % 2 == 0:
                    group1.add(node)    # group 1: even, small
                else:
                    group2.add(node)    # group 2: odd, small
            else:
                if node % 2 == 0:
                    group3.add(node)    # group 3: even, large
                else:
                    group4.add(node)    # group 4: odd, large

        return group0, group1, group2, group3, group4

    @abstractmethod
    def find_route(self) -> Result:
        pass

    def log(self, **kwargs) -> None:
        record: Dict[str: Any] = {}
        record.update({
            'csv_path': self.csv_path, 
            'n': self.n, 
            'method': self.__class__.__name__
        })
        record.update(kwargs)
        print(record)


