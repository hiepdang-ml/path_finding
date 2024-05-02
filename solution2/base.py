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

    def compute_D(self, path: Path) -> float:
        return self.distances[path[:-1], path[1:]].sum()

    def compute_delta(self, path: Path) -> float:
        distances: NDArray[np.float32] = self.distances[path[:-1], path[1:]]
        return distances.max() - distances.min()
    
    def compute_cost(self, path: Path, log: bool = False) -> Cost:
        delta: float = self.compute_delta(path)
        D: float = self.compute_D(path)
        cost: float = self.n * self.distances.max() * delta + D
        if log:
            print(f"Path : {'-'.join(map(str, path))}")
            print(f'Total distances : {D}')
        return cost
    
    def is_feasible_path(self, path: Path) -> bool:
        if path[0] != 0:
            return False
        if len(path) == self.n + 2 and path[-1] != self.n + 1:    # complete path
            return False
        if len(path) == self.n + 2:
            m: int = self.n + 1
        else:
            m: int = len(path)
        for i in range(m-1, 1, -1):
            if (
                path[i-1] % 2 == 0 and path[i] % 2 == 1 and path[i-1] < self.n // 2
            ) or (
                path[i-1] % 2 == 1 and path[i] % 2 == 0 and path[i-1] >= self.n // 2
            ): 
                return False
        return True

    def generate_all_feasible_paths(self) -> Iterator[Path]:
        def backtrack(path):
            if path[-1] == self.n + 1:  # completed
                yield path
                return
            for node in self.find_allowed_nodes(traveled_path=path):
                yield from backtrack(path + [node])

        yield from backtrack([0])  # Start with an empty path

    def get_random_feasible_paths(self, n: int, seed: Optional[int] = None) -> Set[Path]:
        random.seed(seed)
        random_paths: Set[Path] = set()
        random_ints: List[int] = list(range(1, self.n + 1))
        while len(random_paths) < n:
            random.shuffle(random_ints)
            random_path: Path = tuple([0] + random_ints + [self.n + 1])
            if self.is_feasible_path(random_path):
                random_paths.add(random_path)
        return random_paths
    
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
    
    def find_allowed_nodes(self, traveled_path: List[int]) -> Set[int]:
        if len(traveled_path) == self.n + 1:
            return {self.n + 1}
        
        at_node = traveled_path[-1]
        allowed_nodes: Set[int] = {
            node 
            for node in range(self.n + 1) 
            if node not in traveled_path 
                and self.is_feasbile_transition(from_node=at_node, to_node=node)
        }
        return allowed_nodes
    
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
    def find_path(self) -> Result:
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

    def to_file(self, path: Path) -> None:
        D: float = self.compute_D(path)
        delta: float = self.compute_delta(path)
        cost: float = self.compute_cost(path)
        with open(f'result{self.n}.txt', mode='w') as file:
            file.write(f'Route: {"-".join(map(str, path))}\n')
            file.write(f'Total Distance: {D}\n')
            file.write(f'Delta Value: {delta}\n')
            file.write(f'Total Cost: {cost}\n')

