from typing import List, Tuple, Set
import argparse
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
        q0: float,
        local_decay: float,
        global_decay: float, 
    ) -> None:

        super().__init__(csv_path)
        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.alpha: float = alpha
        self.beta: float = beta
        self.q0: float = q0
        self.local_decay: float = local_decay
        self.global_decay: float = global_decay
        self.min_pheromone: float = 1e-6

        assert alpha >= 0
        assert beta >= 0
        assert 0 <= q0 <= 1
        assert 0 <= local_decay <= 1
        assert 0 <= global_decay <= 1

        # create attractiveness and pheremone matrix, [i, j] = node_i -> node_j
        self.__pheromone_matrix: NDArray[np.float32] = self.__create_initial_pheromone_matrix()
        self.__attractiveness_matrix: NDArray[np.float32] = np.zeros_like(self.distances, dtype=np.float32)
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                if self.is_feasbile_transition(from_node=i, to_node=j):
                    self.__attractiveness_matrix[i, j] = self.attractiveness_func(
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
    
    def update_pheromones(self, new_pheromones: NDArray[np.float32], decay: float) -> None:
        assert self.__pheromone_matrix.shape == new_pheromones.shape
        self.__pheromone_matrix = np.maximum(
            self.__pheromone_matrix * (1 - decay) + decay * new_pheromones,
            self.min_pheromone,
        )

    def __create_initial_pheromone_matrix(self):
        return np.ones_like(a=self.distances, dtype=np.float32) * self.min_pheromone

    def reset_pheromones(self) -> None:
        self.__pheromone_matrix = self.__create_initial_pheromone_matrix()

    def attractiveness_func(self, distance: float) -> float:
        estimated_D: float = distance * (self.n + 1)
        estimated_delta: float = distance
        estimated_cost: float = (
            self.n * estimated_delta * self.distances.max() + estimated_D
        )
        return 100 * (self.n + 1) ** 2 / (estimated_cost + 1e-6)
    
    def pheromone_func(self, cost: Cost) -> float:
        return 100 * (self.n + 1) ** 2 / (cost + 1e-6)
    
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
    
    def compute_importances(self, at_node: int, allowed_nodes: List[int]) -> Tuple[
        NDArray[np.float32], NDArray[np.float32]
    ]:
        assert len(allowed_nodes) > 0
        pheromones: NDArray[np.float32] = self.pheromone_matrix[at_node, allowed_nodes]
        attractivenesses: NDArray[np.float32] = self.attractiveness_matrix[at_node, allowed_nodes]
        pheromone_importances: NDArray[np.float32] = np.power(pheromones, self.alpha)
        attractiveness_importances: NDArray[np.float32] = np.power(attractivenesses, self.beta)
        # print(f'w_pheromones: {pheromone_importances}')
        # print(f'w_attractivenesses: {attractiveness_importances}')
        # print('--')
        return pheromone_importances, attractiveness_importances

    def compute_probabilities(self, at_node: int, allowed_nodes: List[int]) -> NDArray[np.float32]:
        assert len(allowed_nodes) > 0
        pheromone_importances, attractiveness_importances = self.compute_importances(at_node, allowed_nodes)
        # print(f'pheromone_importances: {pheromone_importances}')
        # print(f'attractiveness_importances: {attractiveness_importances}')
        # print('-----')
        numerators: NDArray[np.float32] = pheromone_importances * attractiveness_importances
        denominator: float = numerators.sum()
        return numerators / denominator
    
    def compute_combined_importances(self, at_node: int, allowed_nodes: List[int]) -> NDArray[np.float32]:
        assert len(allowed_nodes) > 0
        pheromone_importances, attractiveness_importances = self.compute_importances(at_node, allowed_nodes)
        return pheromone_importances * attractiveness_importances

    def select_next_node(self, at_node: int, allowed_nodes: List[int]) -> int:
        assert len(allowed_nodes) > 0
        if random.random() < self.q0:
            combined_importances: NDArray[np.float32]
            combined_importances = self.compute_combined_importances(at_node, allowed_nodes)
            next_node: int = allowed_nodes[combined_importances.argmax()]
        else:
            probabilities: NDArray[np.float32] = self.compute_probabilities(at_node, allowed_nodes)
            next_node: int = random.choices(population=allowed_nodes, weights=probabilities, k=1).pop()

        return next_node

    def compute_local_deposited_pheromones(self, route: Route) -> NDArray[np.float32]:
        deposited_pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = route[:-1]
        to_nodes: List[int] = route[1:]
        deposited_pheromone_matrix[from_nodes, to_nodes] = self.min_pheromone
        return deposited_pheromone_matrix
    
    def compute_global_deposited_pheromones(self, cost: Cost, route: Route) -> NDArray[np.float32]:
        deposited_pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = route[:-1]
        to_nodes: List[int] = route[1:]
        deposited_pheromone_matrix[from_nodes, to_nodes] = self.pheromone_func(cost)
        return deposited_pheromone_matrix
    
    def one_ant_run(self) -> Tuple[Cost, Route, NDArray[np.float32]]:
        while True:
            traveled_route: List[int] = [0]
            while len(traveled_route) < self.n + 2:
                allowed_nodes: List[int] = list(self.find_allowed_nodes(traveled_route=traveled_route))
                if len(allowed_nodes) == 0: # run into a dead-end -> retry
                    break

                at_node: int = traveled_route[-1]
                next_node: int = self.select_next_node(at_node=at_node, allowed_nodes=allowed_nodes)
                # make move
                traveled_route.append(next_node)

            else:
                route: Route = tuple(traveled_route)
                # local pheromone deposit
                cost: Cost = self.compute_cost(route)
                local_pheromone_matrix: NDArray[np.float32] = self.compute_local_deposited_pheromones(route)
                return cost, route, local_pheromone_matrix

    @track_time
    def find_route(self) -> Result:
        # reset pheromone matrix
        self.reset_pheromones()
        # initialize best cost
        global_best_cost: Cost = float('inf')
        
        for i in range(self.n_iterations):
            ant_results: List[Tuple[Cost, Route, NDArray[np.float32]]] = [
                self.one_ant_run() for _ in range(self.n_ants)
            ]
            assert len(ant_results) == self.n_ants

            # local pheromone update
            for ant_result in ant_results:
                print(f'Evaluating: {ant_result[1]}')
                # print('local update')
                # print(ant_result[2])
                self.update_pheromones(new_pheromones=ant_result[2], decay=self.local_decay)

            # global pheromone update
            # print('global update')
            local_solution: Tuple[Cost, Route, NDArray[np.float32]] = min(ant_results, key=lambda x: x[0])
            local_best_cost, local_best_route, _ = local_solution
            global_pheromone_matrix = self.compute_global_deposited_pheromones(cost=local_best_cost, route=local_best_route)
            # print(global_pheromone_matrix)
            # print('-----')
            self.update_pheromones(new_pheromones=global_pheromone_matrix, decay=self.global_decay)

            if local_best_cost < global_best_cost:
                # update global solution
                global_best_cost = local_best_cost
                global_best_route = local_best_route
            
            # Log the best result so far
            print(f'Iteration: {i}/{self.n_iterations}')
            print(f'Best route found so far: {global_best_route}, best_cost: {global_best_cost}')
            print(f'---------------')

        return global_best_cost, global_best_route



if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser(description='Run Ant Colony Optimization Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_ants', '-n', type=int, default=500, help='Number of ants at each iteration')
    parser.add_argument('--n_iterations', '-N', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--alpha', '-a', type=float, default=0.2, help='Relative importance of pheromone (exploitation)')
    parser.add_argument('--beta', '-b', type=float, default=3., help='Relative importance of attractiveness (exploration)')
    parser.add_argument('--q0', type=float, default=0.05, help='Relative importance of exploration (0.) vs. exploitation (1.)')
    parser.add_argument('--local_decay', '-ld', type=float, default=0.2, help='Local pheromone decay rate (exploration)')
    parser.add_argument('--global_decay', '-gd', type=float, default=0.4, help='Global pheromone decay rate (exploration)')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = AntSystem(**vars(args))
    r: Result = solver.find_route()
    print(f'Found solution: {r}')

# Best route found so far: (0, 6, 5, 7, 1, 2, 8, 9, 3, 4, 10, 11), best_cost: 27575.20194778446
