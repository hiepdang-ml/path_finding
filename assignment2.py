from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Callable, TypeAlias, Iterator, Any
from functools import cached_property, lru_cache, wraps
from itertools import permutations
import heapq
import random
import time

import numpy as np
from numpy.typing import NDArray

# UTILITIES:

Route: TypeAlias = Tuple[int, ...]
Cost: TypeAlias = float
Result: TypeAlias = Tuple[Cost, Route]

def track_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0: float = time.time()
        result: Any = func(self, *args, **kwargs)
        self.duration = time.time() - t0
        return result
    return wrapper


# IMPLEMENTATION:

class Solver(ABC):

    def __init__(self, csv_path: str) -> None:
        self.csv_path: str = csv_path
        
        with open(file=csv_path, mode='r') as file:
            next(file)      # skip first row
            data: List[str] = file.readlines()
            pre_process = lambda s: [
                float(x) for x in s.replace('\n','').split(',')[-2:]
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

    def compute_D(self, route: Route) -> float:
        return self.distances[route[:-1], route[1:]].sum()

    def compute_delta(self, route: Route) -> float:
        distances: NDArray[np.float32] = self.distances[route[:-1], route[1:]]
        return distances.max() - distances.min()
    
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
    
    def generate_all_feasible_routes(self) -> Iterator[Route]:   # expensive
        all_routes: Iterator[Route] = map(
            lambda x: (0,) + x + (self.n + 1,), 
            permutations(iterable=range(1, self.n + 1), r=self.n)
        )
        for route in all_routes:
            if self.is_feasible_route(route):
                yield route

    def get_random_feasible_routes(self, n: int, seed: int) -> Set[Route]:
        random.seed(seed)
        random_routes: Set[Route] = set()
        random_ints: List[int] = list(range(1, self.n + 1))
        while len(random_routes) < n:
            random.shuffle(random_ints)
            random_route: Route = tuple([0] + random_ints + [self.n + 1])
            if self.is_feasible_route(random_route):
                random_routes.add(random_route)
        return random_routes


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


class BruteForce(Solver):

    @track_time
    def find_route(self) -> Result:             # expensive
        feasible_routes: Iterator[Route] = self.generate_all_feasible_routes()
        results: List[Result] = []
        for route in feasible_routes:
            print(f'Evaluating: {route}')
            results.append((self.compute_cost(route), route))
        return min(results, key=lambda x: x[0])


class AStar(Solver):

    def __init__(
        self, 
        csv_path: str, 
        n_random_routes: int = 100,
        h_coeff: float = 0.8,
        seed: int = +1_347_247_9378,
    ) -> None:
        super().__init__(csv_path)
        self.n_random_routes: int = n_random_routes
        self.h_coeff: float =h_coeff
        self.seed: int = seed

    @track_time
    def find_route(self) -> Result:
        random_routes: Set[Route] = self.get_random_feasible_routes(n=self.n_random_routes, seed=self.seed)
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


class GeneticAlgorithm(Solver):

    def __init__(
        self, 
        csv_path: str, 
        population_size: int = 1000,
        tolerance: int = 100,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        seed: int = +1_347_247_9378,
    ):
        super().__init__(csv_path)
        self.population_size: int = population_size
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate
        self.tolerance: int = tolerance

        self.seed: int = seed
        random.seed(seed)
        
        self.population: List[Result] = [
            (self.compute_cost(route), route)
            for route in self.get_random_feasible_routes(n=self.population_size, seed=seed)
        ]
        self.population = sorted(self.population, key=lambda x: x[0])
        self.bred_offsprings: Set[Result] = set()

    def are_same_group_nodes(self, node1: int, node2: int) -> bool:
        if node1 % 2 != node2 % 2:
            return False
        if node1 < self.n // 2 and node2 >= self.n // 2:
            return False
        if node1 >= self.n // 2 and node2 < self.n // 2:
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
                    group1.add(node)
                else:
                    group2.add(node)
            else:
                if node % 2 == 0:
                    group3.add(node)
                else:
                    group4.add(node)
        return group0, group1, group2, group3, group4

    def selection(self) -> Tuple[List[Route], List[Route]]:
        fitnesses = np.array(
            [individual[0] for individual in self.population],  # fitness function = cost function
            dtype=np.float32,
        )
        population_routes: List[Route] = [individual[1] for individual in self.population]
        probabilities: NDArray[np.float32] = fitnesses / fitnesses.sum()
        parent1s: List[Route] = random.choices(
            population=population_routes,
            k=len(population_routes) // 2,
            weights=probabilities
        )
        parent2s: List[Route] = random.choices(
            population=population_routes, 
            k=len(population_routes) // 2, 
            weights=probabilities
        )
        return parent1s, parent2s
    
    # TODO: implement standard position-preserving crossover
    def crossover(self, parent1s: List[Route], parent2s: List[Route]) -> List[Route]:
        offsprings: List[Route] = []
        for parent1, parent2 in zip(parent1s, parent2s):
            # Start from parents
            offspring1: List[int] = list(parent1)
            offspring2: List[int] = list(parent2)
            
            for i in range(1, self.n + 1):
                node1: int = offspring1[i]
                node2: int = offspring2[i]
                
                if random.random() < self.crossover_rate and self.are_same_group_nodes(node1=node1, node2=node2):
                    offspring1[offspring1.index(node2)] = node1
                    offspring1[i] = node2
                    offspring2[offspring2.index(node1)] = node2
                    offspring2[i] = node1

            offsprings.extend([tuple(offspring1), tuple(offspring2)])
        
        return offsprings

    # TODO: implement standard position-preserving mutation
    def mutation(self, offsprings: List[Route]) -> List[Route]:
        mutated_offsprings: List[Route] = []
        
        for offspring in offsprings:
            mutated_offspring: Route = offspring[:]

            for group in self.node_groups[1:]:  # skip group 0
                if random.random() < self.mutation_rate:
                    to_permute: List[int] = [node for node in offspring if node in group]
                    random.shuffle(to_permute)
                    mutated_offspring: Route = tuple(
                        to_permute.pop() if node in group else node for node in mutated_offspring
                    )
            
            mutated_offsprings.append(mutated_offspring)
        
        return mutated_offsprings

    @track_time
    def find_route(self) -> Result:
        best_cost: float; best_route: Route
        best_cost, best_route = self.population[0]
        n_unimprovements: int = 0

        while True:
            print(f'Number of bred offsprings: {len(self.bred_offsprings)}')
            print(f'Best (complete) route found so far: {best_route}, best_cost: {best_cost}')
            parent1s: List[Route]; parent2s: List[Route]
            parent1s, parent2s = self.selection()
            offsprings: List[Route] = self.crossover(parent1s, parent2s)
            offsprings: List[Route] = self.mutation(offsprings)
            # offsprings: Set[Route] = set(offsprings) - set(individual[1] for individual in self.population)
            offsprings: Set[Route] = set(offsprings) - set(individual[1] for individual in self.bred_offsprings)
            print(f'Number of new offsprings: {len(offsprings)}')
            self.bred_offsprings.update(offsprings)
            self.population: List[Result] = self.population + [
                (self.compute_cost(offspring), offspring)
                for offspring in offsprings
            ]
            self.population: List[Result] = sorted(
                self.population, key=lambda x: x[0]
            )[:self.population_size]  # fixed population size
            cost, route = self.population[0]
        
            if cost < best_cost:
                best_cost = cost
                best_route = route
                n_unimprovements = 0
            else:
                n_unimprovements += 1
                print(
                    f'Could not improve result: '
                    f'n_unimprovements/tolerance = {n_unimprovements}/{self.tolerance}'
                )
                if n_unimprovements >= self.tolerance:
                    break
            
            print('---------------')
        
        return best_cost, best_route



if __name__ == '__main__':

    # t0: float = time.time()
    # solver: Solver = AStar(
    #     csv_path='assignments/Assessment_II/data/I20.csv',
    #     n_random_routes=100, 
    #     h_coeff=0.8,
    # )
    # result: Result = solver.find_route()
    # solver.log(
    #     n_random_routes=solver.n_random_routes, 
    #     h_coeff=solver.h_coeff,
    #     solution=result[1],
    #     cost=result[0],
    #     D=solver.compute_D(route=result[1]),
    #     delta=solver.compute_delta(route=result[1]),
    #     duration=solver.duration,
    # )


    # solver = EvolutionaryAlgorithm(
    #     csv_path='assignments/Assessment_II/data/I20.csv',
    #     population_size=100,
    #     n_generations=100,
    #     mutation_rate=0.01,
    # )
    # for i in range(50):
    #     p1, p2 = solver.get_random_feasible_routes(n=2, seed=i)
    #     o1, o2 = solver.crossover(parent1s=[p1], parent2s=[p2])
    #     for r in [p1, p2, o1, o2]:
    #         print(r)
    #         print(solver.is_feasible_route(r))
    #         print('---')
    #     print('=========')


    solver = GeneticAlgorithm(
        csv_path='assignments/Assessment_II/data/I20.csv',
        population_size=1000,
        tolerance=1000,
        crossover_rate=0.5,
        mutation_rate=0.9,
        seed=+1_347_247_9378,
    )
    r = solver.find_route()




