from typing import List, Tuple, Set
import argparse
import random

import numpy as np
from numpy.typing import NDArray

from utils.functional import track_time
from utils.type_alias import Path, Cost, Result
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
        decay: float,
        reinforce: float, 
    ) -> None:

        super().__init__(csv_path)
        if n_ants is None:
            self.n_ants: int = self.n + 1   # A common approach: approximately equal to the number of decisions
        else:
            self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.alpha: float = alpha
        self.beta: float = beta
        self.q0: float = q0
        self.decay: float = decay
        self.reinforce: float = reinforce
        self.min_pheromone: float = 1. + 1e-6

        assert alpha >= 0
        assert beta >= 0
        assert 0 <= q0 <= 1
        assert 0 <= decay <= 1
        assert 0 <= reinforce <= 1

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
        # indices where new pheromones have been deposited:
        nonzero_indices: NDArray[np.bool_] = np.nonzero(new_pheromones)
        self.__pheromone_matrix[nonzero_indices] = (
            self.__pheromone_matrix[nonzero_indices] * (1 - decay)
            + new_pheromones[nonzero_indices] * decay
        )

    def __create_initial_pheromone_matrix(self):
        return np.ones_like(a=self.distances, dtype=np.float32) * self.min_pheromone

    def reset_pheromones(self) -> None:
        self.__pheromone_matrix = self.__create_initial_pheromone_matrix()

    def attractiveness_func(self, distance: float) -> float:
        estimated_D: float = distance * (self.n + 1)
        estimated_delta: float = distance / 2
        estimated_cost: float = (
            self.n * self.distances.max() * estimated_delta + estimated_D
        )
        return 100 * (self.n + 1) ** 2 / (estimated_cost + 1e-6) + 1
    
    def pheromone_func(self, cost: Cost) -> float:
        return 100 * (self.n + 1) ** 2 / (cost + 1e-6) + 1
    
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
    
    def compute_importances(self, at_node: int, allowed_nodes: List[int]) -> Tuple[
        NDArray[np.float32], NDArray[np.float32]
    ]:
        assert len(allowed_nodes) > 0
        pheromones: NDArray[np.float32] = self.pheromone_matrix[at_node, allowed_nodes]
        attractivenesses: NDArray[np.float32] = self.attractiveness_matrix[at_node, allowed_nodes]
        pheromone_importances: NDArray[np.float32] = np.power(pheromones, self.alpha)
        attractiveness_importances: NDArray[np.float32] = np.power(attractivenesses, self.beta)
        return pheromone_importances, attractiveness_importances

    def compute_probabilities(self, at_node: int, allowed_nodes: List[int]) -> NDArray[np.float32]:
        assert len(allowed_nodes) > 0
        # print(f'at_node: {at_node}')
        # print(f'allowed_nodes: {allowed_nodes}')
        pheromone_importances, attractiveness_importances = self.compute_importances(at_node, allowed_nodes)
        # print(f'pheromone_importance: {pheromone_importances}')
        # print(f'attractiveness_importances: {attractiveness_importances}')
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
            # print(f'probabilities: {probabilities}')
            next_node: int = random.choices(population=allowed_nodes, weights=probabilities, k=1).pop()
            # print(f'select node: {next_node}')
            # print('-----')

        return next_node

    def compute_local_deposited_pheromones(self, path: Path) -> NDArray[np.float32]:
        deposited_pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = path[:-1]
        to_nodes: List[int] = path[1:]
        deposited_pheromone_matrix[from_nodes, to_nodes] = self.min_pheromone
        return deposited_pheromone_matrix
    
    def compute_global_deposited_pheromones(self, cost: Cost, path: Path) -> NDArray[np.float32]:
        deposited_pheromone_matrix: NDArray[np.float32] = np.zeros_like(a=self.distances, dtype=np.float32)
        from_nodes: List[int] = path[:-1]
        to_nodes: List[int] = path[1:]
        deposited_pheromone_matrix[from_nodes, to_nodes] = self.pheromone_func(cost)
        return deposited_pheromone_matrix
    
    def one_ant_run(self) -> Result:
        while True:
            traveled_path: List[int] = [0]
            while len(traveled_path) < self.n + 2:
                allowed_nodes: List[int] = list(self.find_allowed_nodes(traveled_path=traveled_path))
                if len(allowed_nodes) == 0: # run into a dead-end -> retry
                    break

                at_node: int = traveled_path[-1]
                next_node: int = self.select_next_node(at_node=at_node, allowed_nodes=allowed_nodes)
                # make move
                traveled_path.append(next_node)
            else:
                path: Path = tuple(traveled_path)
                # local pheromone deposit
                cost: Cost = self.compute_cost(path)
                local_pheromone_matrix: NDArray[np.float32] = self.compute_local_deposited_pheromones(path)
                # print(f'Evaluating: {path}, cost: {cost}')
                # local pheromone update
                self.update_pheromones(new_pheromones=local_pheromone_matrix, decay=self.decay)
                return cost, path

    @track_time
    def find_path(self) -> Result:
        # reset pheromone matrix
        self.reset_pheromones()
        # initialize best cost
        global_best_cost: Cost = float('inf')
        
        for i in range(self.n_iterations):
            # occasionally reset pheromones to break out of local optima
            if random.random() <= 0.01:
                print('Reset global pheromones matrix')
                self.reset_pheromones()

            ant_results: List[Result] = [self.one_ant_run() for ant_id in range(self.n_ants)]
            assert len(ant_results) == self.n_ants
            # global pheromone update
            print(f'Iteration: {i + 1}/{self.n_iterations}')
            print(f'Average cost: {sum([result[0] for result in ant_results]) / self.n_ants}')
            local_solution: Result = min(ant_results, key=lambda x: x[0])
            local_best_cost, local_best_path = local_solution
            print(f'Best path found in the iteration: {local_best_path}, best cost: {local_best_cost}')
            global_pheromone_matrix = self.compute_global_deposited_pheromones(cost=local_best_cost, path=local_best_path)

            if local_best_cost < global_best_cost:
                # update global solution
                global_best_cost = local_best_cost
                global_best_path = local_best_path
                # global update with high decay
                self.update_pheromones(new_pheromones=global_pheromone_matrix, decay=self.reinforce)

            # Log the best result so far
            print(f'Best path found so far:{" "*11}{global_best_path}, best cost: {global_best_cost}')
            print(f'---------------')

        return global_best_cost, global_best_path



if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser(description='Run Ant Colony Optimization Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--n_ants', '-n', type=int, default=100, help='Number of ants at each iteration')
    parser.add_argument('--n_iterations', '-N', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--alpha', '-a', type=float, default=4., help='Relative importance of pheromone (for exploitation)')
    parser.add_argument('--beta', '-b', type=float, default=4., help='Relative importance of attractiveness (for exploration)')
    parser.add_argument('--q0', type=float, default=0.1, help='Relative importance of exploration (0.) vs. exploitation (1.)')
    parser.add_argument('--decay', '-d', type=float, default=1e-6, help='Local pheromone decay rate (for exploration)')
    parser.add_argument('--reinforce', '-r', type=float, default=0.7, help='Global pheromone decay rate (for exploitation)')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = AntSystem(**vars(args))
    r: Result = solver.find_path()
    print(f'Found solution: {r}')
    p = solver.pheromone_matrix
    a = solver.attractiveness_matrix
    pa = p * a
    pa[0] / pa[0].sum()


# Best path found so far: (0, 13, 5, 6, 10, 12, 11, 9, 16, 14, 4, 8, 2, 20, 18, 15, 7, 17, 19, 3, 1, 21), best cost: 72456.09841918945
# solver.compute_cost(tuple(map(int,'0-14-1-9-20-3-4-2-8-16-12-6-18-11-15-5-10-13-7-17-19-21'.split('-'))))