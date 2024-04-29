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
        p0: float,
        local_decay: float,
        global_decay: float, 
    ) -> None:

        super().__init__(csv_path)
        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.alpha: float = alpha
        self.beta: float = beta
        self.p0: float = p0
        self.local_decay: float = local_decay
        self.global_decay: float = global_decay
        self.min_pheromone: float = 1e-3

        assert alpha > 0
        assert beta >= 1
        assert 0 <= p0 <= 1
        assert 0 <= local_decay <= 1
        assert 0 <= global_decay <= 1

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
            self.__pheromone_matrix * (1 - self.global_decay) 
            + new_pheromones
        )

    def __create_initial_pheromone_matrix(self):
        return np.ones_like(a=self.distances, dtype=np.float32) * self.min_pheromone

    def reset_pheromones(self) -> None:
        self.__pheromone_matrix = self.__create_initial_pheromone_matrix()

    @staticmethod
    def attractiveness_func(distance: float) -> float:
        P: float = 100.
        return P / (distance + 1e-3)
    
    def pheromone_func(self, cost: Cost) -> float:
        Q: float = 100.
        return max(Q * (self.n + 1) / (cost + 1e-3), self.min_pheromone)
    
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
    
    def compute_importances(self, traveled_route: List[int]) -> Tuple[
        int, List[int], NDArray[np.float32], NDArray[np.float32]
    ]:
        at_node: int = traveled_route[-1]
        allowed_nodes: List[int] = list(self.find_allowed_nodes(traveled_route=traveled_route))
        if len(allowed_nodes) == 0:
            return [], []
        
        pheromones: NDArray[np.float32] = self.pheromone_matrix[at_node, allowed_nodes]
        attractivenesses: NDArray[np.float32] = self.attractiveness_matrix[at_node, allowed_nodes]
        pheromone_importances: NDArray[np.float32] = np.power(pheromones, self.alpha)
        attractiveness_importances: NDArray[np.float32] = np.power(attractivenesses, self.beta)
        # print(f'w_pheromones: {pheromone_importances} - w_attractivenesses: {attractiveness_importances}')
        return at_node, allowed_nodes, pheromone_importances, attractiveness_importances

    def compute_probabilities(self, traveled_route: List[int]) -> Tuple[List[int], NDArray[np.float32]]:
        _, allowed_nodes, pheromone_importances, attractiveness_importances = self.compute_importances(
            traveled_route=traveled_route
        )
        numerators: NDArray[np.float32] = pheromone_importances * attractiveness_importances
        denominator: float = numerators.sum()
        return allowed_nodes, numerators / denominator
    
    def compute_combined_importances(self, traveled_route: List[int]) -> Tuple[List[int], NDArray[np.float32]]:
        _, allowed_nodes, pheromone_importances, attractiveness_importances = self.compute_importances(
            traveled_route=traveled_route
        )
        return allowed_nodes, pheromone_importances * attractiveness_importances

    def select_next_node(self, traveled_route: List[int]) -> int:
        allowed_nodes: List[int]
        if random.random() < self.p0:
            combined_importances: NDArray[np.float32]
            allowed_nodes, combined_importances = self.compute_combined_importances(traveled_route)
            next_node: int = allowed_nodes[combined_importances.argmax()]
        else:
            probabilities: NDArray[np.float32]
            allowed_nodes, probabilities = self.compute_probabilities(traveled_route)
            next_node: int = random.choices(population=allowed_nodes, weights=probabilities, k=1).pop()

        return next_node        

    def compute_local_deposited_pheromones(self, route: Route) -> NDArray[np.float32]:
        pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = route[:-1]
        to_nodes: List[int] = route[1:]
        pheromone_matrix[from_nodes, to_nodes] = self.min_pheromone
        return pheromone_matrix
    
    def compute_global_deposited_pheromones(self, cost: Cost, route: Route) -> NDArray[np.float32]:
        pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = route[:-1]
        to_nodes: List[int] = route[1:]
        pheromone_matrix[from_nodes, to_nodes] = self.pheromone_func(cost)





    def one_ant_run(self, ant_id: int) -> Tuple[Cost, Route, NDArray[np.float32]]:
        while True:
            traveled_route: List[int] = [0]
            while len(traveled_route) < self.n + 2:
                # node selection
                allowed_nodes, probabilities = self.compute_probabilities(traveled_route=traveled_route)
                if len(allowed_nodes) == 0:   # run into a dead-end -> start over
                    break
                next_node: int = AntSystem.select_next_node(
                    allowed_nodes=allowed_nodes, 
                    probabilities=probabilities
                )
                # make move
                traveled_route.append(next_node)

            if len(traveled_route) == self.n + 2:
                route: Route = tuple(traveled_route)
                # deposit pheromones
                cost: Cost = self.compute_cost(route)
                deposited_pheromone = self.pheromone_func(cost)
                deposited_pheromone_matrix: NDArray = np.zeros_like(a=self.pheromone_matrix, dtype=np.float32)
                deposited_pheromone_matrix[route[:-1], route[1:]] = deposited_pheromone
                return cost, route, deposited_pheromone_matrix

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
            ant_results: List[Tuple[Cost, Route, NDArray[np.float32]]] = [
                self.one_ant_run(ant_id) for ant_id in range(self.n_ants)
            ]
            assert len(ant_results) == self.n_ants

            # update best solution
            solution: Tuple[Cost, Route, NDArray[np.float32]] = min(ant_results, key=lambda x: x[0])
            cost, route = solution[:2]
            if cost < best_cost:
                best_cost = cost
                best_route = route
            
            # update pheromone
            deposited_pheromones = np.array([result[2] for result in ant_results], dtype=np.float32)
            assert deposited_pheromones.shape == (self.n_ants, self.n + 2, self.n + 2)
            self.update_pheromones(new_pheromones=deposited_pheromones.sum(axis=0))
            print(f'Updated pheromone at iteration: {i + 1}')

            # Log the best result so far
            print(f'Best route found so far: {best_route}, best_cost: {best_cost}')
            print('---------------')

        return best_cost, best_route


def main():

    parser = argparse.ArgumentParser(description='Run Ant Colony Optimization Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_ants', '-n', type=int, default=500, help='Number of ants at each iteration')
    parser.add_argument('--n_iterations', '-N', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', '-a', type=float, default=0.8, help='Alpha coefficient')
    parser.add_argument('--beta', '-b', type=float, default=1., help='Beta coefficient')
    parser.add_argument('--evaporation', '-e', type=float, default=0.95, help='Evaporation coefficient')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = AntSystem(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser(description='Run Ant Colony Optimization Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_ants', '-n', type=int, default=500, help='Number of ants at each iteration')
    parser.add_argument('--n_iterations', '-N', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', '-a', type=float, default=0.8, help='Alpha coefficient')
    parser.add_argument('--beta', '-b', type=float, default=1., help='Beta coefficient')
    parser.add_argument('--evaporation', '-e', type=float, default=0.55, help='Evaporation coefficient')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = AntSystem(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')

