from typing import List, Tuple, Set
import argparse
from multiprocessing import Pool, cpu_count
import random

import numpy as np
from numpy.typing import NDArray

from utils.functional import track_time
from utils.type_alias import Route, Cost, Result
from base import Solver


class AntSystem(Solver):

    def __init__(
        self, 
        csv_path: str, 
        n_ants: int, 
        n_iterations: int,
        alpha: float, 
        beta: float, 
        evaporation: float, 
    ) -> None:

        super().__init__(csv_path)
        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.alpha: float = alpha
        self.beta: float = beta
        self.evaporation: float = evaporation

        assert alpha >= 0
        assert beta >= 1
        assert 0 <= evaporation <= 1

        # create attractiveness and pheremone matrix, [i, j] = node_i -> node_j
        self.__pheromone_matrix: NDArray[np.float32] = self.__create_initial_pheromone_matrix()
        self.__attractiveness_matrix: NDArray[np.float32] = np.zeros_like(self.distances, dtype=np.float32)
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                if self.is_feasbile_transition(from_node=i, to_node=j):
                    self.__attractiveness_matrix[i, j] = AntSystem.attractiveness_func(
                        distance=self.distances[i, j]
                    )

    # read-only
    @property
    def attractiveness_matrix(self) -> NDArray[np.float32]:
        # define getter to prevent direct value assignments
        return self.__attractiveness_matrix
    
    # read-only
    @property
    def pheromone_matrix(self) -> NDArray[np.float32]:
        # define getter to prevent direct value assignments.
        # it should only be updated by method `update_pheromone` 
        # or reset by `reset_pheromones`
        return self.__pheromone_matrix
    
    def update_pheromones(self, new_pheromones: NDArray[np.float32]) -> None:
        assert self.__pheromone_matrix.shape == new_pheromones.shape
        self.__pheromone_matrix = (
            self.__pheromone_matrix * (1 - self.evaporation) 
            + new_pheromones
        )

    def __create_initial_pheromone_matrix(self):
        return np.ones_like(self.distances, dtype=np.float32) * 1e-10

    def reset_pheromones(self) -> None:
        self.__pheromone_matrix = self.__create_initial_pheromone_matrix()

    @staticmethod
    def attractiveness_func(distance: float) -> float:
        return 1 / (distance + 1e-10)
    
    @staticmethod
    def pheromone_func(distance: float) -> float:
        return 1 / (distance + 1e-10)
    
    def find_allowed_nodes(self, traveled_route: List[int]) -> Set[int]:
        if len(traveled_route) == self.n + 1:
            return {self.n + 1}
        
        at_node = traveled_route[-1]
        allowed_nodes: Set[int] = {
            node 
            for node in range(self.n + 1) 
            if node not in traveled_route 
                and self.is_feasbile_transition(from_node=at_node, to_node=node)
        }
        return allowed_nodes
    
    def compute_probabilities(self, traveled_route: List[int]) -> Tuple[List[int], List[float]]:
        at_node: int = traveled_route[-1]
        allowed_nodes: List[int] = list(self.find_allowed_nodes(traveled_route=traveled_route))
        if len(allowed_nodes) == 0:
            return [], []
        
        pheromones: NDArray[np.float32] = self.pheromone_matrix[at_node, allowed_nodes]
        attractivenesses: NDArray[np.float32] = self.attractiveness_matrix[at_node, allowed_nodes]
        numerators: NDArray[np.float32] = np.power(attractivenesses, self.alpha) * np.power(pheromones, self.beta)
        denominator: float = numerators.sum()
        probabilities: List[float] = (numerators / denominator).tolist()
        return allowed_nodes, probabilities
    
    @staticmethod
    def select_next_node(allowed_nodes: List[int], probabilities: List[float]) -> int:
        return random.choices(population=allowed_nodes, weights=probabilities, k=1).pop()
    
    def one_ant_run(self, ant_id: int) -> Tuple[Route, NDArray[np.float32]]:
        deposited_pheromone_matrix: NDArray = np.zeros_like(a=self.pheromone_matrix, dtype=np.float32)
        traveled_route: List[int] = [0]
        for t in range(self.n + 1):     # timestep
            at_node: int = traveled_route[-1]
            # node selection
            allowed_nodes, probabilities = self.compute_probabilities(traveled_route=traveled_route)
            if len(allowed_nodes) == 0:   # run into a dead-end
                break
            next_node: int = AntSystem.select_next_node(
                allowed_nodes=allowed_nodes, 
                probabilities=probabilities
            )
            # make move
            traveled_route.append(next_node)
            # deposit pheromones
            deposited_pheromone_matrix[at_node, next_node] += self.pheromone_func(
                distance=self.distances[at_node, next_node]
            )
        
        if len(traveled_route) == self.n + 2:
            return tuple(traveled_route), deposited_pheromone_matrix
        else:
            return self.one_ant_run(ant_id)

    @track_time
    def find_route(self) -> Result:
        # reset pheromone matrix
        self.reset_pheromones()
        # initialize best cost
        best_cost: Cost = float('inf')
        
        for i in range(self.n_iterations):
            deposited_pheromones: NDArray[np.float32] = np.zeros(
                shape=(self.n_ants, self.n + 2, self.n + 2),
                dtype=np.float32,
            )
            with Pool(processes=cpu_count()) as pool:
                # we can do multiprocessing because each ant run independently in each iteration.
                # at each iteration, each ant share the same `self.attractiveness_matrix` and 
                # `self.pheromone_matrix`
                ant_results: List[Tuple[Route, NDArray[np.float32]]] = pool.map(
                    func=self.one_ant_run, iterable=range(self.n_ants)
                )
            
            # update best solution
            discovered_routes: List[Result] = [
                (self.compute_cost(result[0]), result[0]) 
                for result in ant_results
            ]
            solution: Result = min(discovered_routes, key=lambda x: x[0])
            cost, route = solution
            if cost < best_cost:
                best_cost = cost
                best_route = route
            
            # update pheromone
            deposited_pheromones = np.array([result[1] for result in ant_results], dtype=np.float32)
            assert deposited_pheromones.shape == (self.n_ants, self.n + 2, self.n + 2)
            self.update_pheromones(new_pheromones=deposited_pheromones.sum(axis=0))
            print(f'Updated pheromone at iteration: {i}')

            # Log the best result so far
            print(f'Best route found so far: {best_route}, best_cost: {best_cost}')
            print('---------------')

        return best_cost, best_route


def main():

    parser = argparse.ArgumentParser(description='Run Ant Colony Optimization Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_ants', '-n', type=int, default=10000, help='Number of ants at each iteration')
    parser.add_argument('--n_iterations', '-N', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help='Alpha coefficient')
    parser.add_argument('--beta', '-b', type=float, default=1., help='Beta coefficient')
    parser.add_argument('--evaporation', '-e', type=float, default=0.5, help='Evaporation coefficient')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = AntSystem(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    main()

