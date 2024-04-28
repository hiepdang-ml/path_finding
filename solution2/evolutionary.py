from typing import List, Tuple, Set
import random

import numpy as np
from numpy.typing import NDArray

from .utils.type_alias import Route, Cost, Result
from .utils.functional import track_time
from .utils.exception import NeedDifferentCrossoverPoints
from .base import Solver


class GeneticAlgorithm(Solver):

    def __init__(
        self, 
        csv_path: str, 
        population_size: int = 1000,
        tolerance: int = 1000,
        seed: int = +1_347_247_9378,
    ):
        super().__init__(csv_path)
        self.population_size: int = population_size
        self.tolerance = tolerance

        self.seed: int = seed
        random.seed(seed)
        
        self.population: List[Result] = [
            (self.compute_cost(route), route)
            for route in self.get_random_feasible_routes(n=self.population_size, seed=seed)
        ]
        self.population: List[Result] = sorted(self.population, key=lambda x: x[0])
        self.bred_offsprings: Set[Route] = set([individual[1] for individual in self.population])
    
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
    
    def custom_ox1_crossover(self, parent1: Route, parent2: Route) -> Route:
        """ 
        Custom OX1 crossover
        """
        p1: List[int] = list(parent1[1: -1])
        p2: List[int] = list(parent2[1: -1])
        assert len(p1) == self.n
        
        while True:
            try:
                start, end = sorted(random.sample(range(1, self.n-1), 2))
                offspring: List[int] = [None] * self.n
                middle_section: List[int] = p1[start:end+1]
                offspring[start:end+1] = middle_section

                left_index: int = start
                right_index: int = end + 1

                # Left fill
                remaining: List[int] = list(reversed([node for node in p2 if node not in offspring]))
                for i in range(left_index-1, -1, -1):
                    for node in remaining:
                        if self.is_feasbile_transition(node, offspring[i+1]):
                            offspring[i] = node
                            remaining.remove(node)
                            break
                    else:
                        # failed to left fill, choose different crossover points
                        raise NeedDifferentCrossoverPoints

                # Right fill
                remaining: List[int] = [node for node in p2 if node not in offspring]
                for i in range(right_index, self.n):
                    for node in remaining:
                        if self.is_feasbile_transition(offspring[i-1], node):
                            offspring[i] = node
                            remaining.remove(node)
                            break
                    else:
                        # failed to right fill, choose different crossover points
                        raise NeedDifferentCrossoverPoints

                # Successfully filled, break the loop
                break 

            except NeedDifferentCrossoverPoints:
                continue

        offspring: Route = tuple([0] + offspring + [self.n + 1])
        assert None not in offspring
        assert self.is_feasible_route(offspring)
        return offspring
    
    def crossover(self, parent1s: List[Route], parent2s: List[Route]) -> List[Route]:
        offsprings: List[Route] = []
        for parent1, parent2 in zip(parent1s, parent2s):
            offspring1: Route = self.custom_ox1_crossover(parent1, parent2)
            offspring2: Route = self.custom_ox1_crossover(parent2, parent1)
            offsprings.extend([offspring1, offspring2])
        return offsprings

    def custom_right_rotation_mutation(self, offspring: Route) -> Route:
        mutated_offspring: List[int] = list(offspring)
        selected_groups: List[Set[int]] = random.sample(
            population=self.node_groups[1:],    # skip group0
            k=random.randint(a=0, b=4)
        )
        for selected_group in selected_groups:
            indices: List[int] = [i for i in range(len(offspring)) if offspring[i] in selected_group]
            new_indices: List[int] = (indices * 2)[-1-len(indices):-1]  # right shift by 1
            for i, j in zip(indices, new_indices):
                mutated_offspring[i] = offspring[j]
        
        mutated_offspring: Route = tuple(mutated_offspring)
        assert self.is_feasible_route(mutated_offspring)
        return mutated_offspring

    def mutation(self, offsprings: List[Route]) -> List[Route]:
        mutated_offsprings: List[Route] = [
            self.custom_right_rotation_mutation(offspring)
            for offspring in offsprings
        ]
        return mutated_offsprings

    @track_time
    def find_route(self) -> Result:
        best_cost: Cost; best_route: Route
        best_cost, best_route = self.population[0]
        no_new_offsprings: int = 0

        while no_new_offsprings < self.tolerance:
            print(f'Number of bred offsprings: {len(self.bred_offsprings)}')
            print(f'Best (complete) route found so far: {best_route}, best_cost: {best_cost}')
            parent1s: List[Route]; parent2s: List[Route]
            parent1s, parent2s = self.selection()
            offsprings: List[Route] = self.crossover(parent1s, parent2s)
            offsprings: List[Route] = self.mutation(offsprings)
            offsprings: Set[Route] = set(offsprings) - self.bred_offsprings
            n_new_offsprings: int = len(offsprings)
            print(f'Number of new offsprings: {n_new_offsprings}')

            if n_new_offsprings == 0:
                no_new_offsprings += 1
                print(f'Could not breed any new offsprings: {no_new_offsprings}/{self.tolerance}')

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

            print('---------------')
        
        return best_cost, best_route
