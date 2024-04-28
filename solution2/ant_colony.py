from typing import List, Tuple, Set
import random

import numpy as np
from numpy.typing import NDArray

from .utils.functional import track_time
from .utils.type_alias import Route, Cost, Result
from .base import Solver


class AntSystem(Solver):

    def __init__(
        self, 
        csv_path: str, 
        n_ants: int, 
        alpha: float, 
        beta: float, 
        evaporation: float, 
        seed: int = +1_347_247_9378
    ) -> None:
        self.n_ants: int = n_ants
        self.alpha: float = alpha
        self.beta: float = beta
        self.evaporation: float = evaporation
        self.seed: int = seed
        super().__init__(csv_path)

        assert alpha >= 0
        assert beta >= 1
        assert 0 <= evaporation <= 1

        # create attractiveness and pheremone matrix, [i, j] = node_i -> node_j
        self.__attractiveness_matrix: NDArray[np.float32] = self.__initialize_attractiveness_matrix()
        self.__pheromone_matrix: NDArray[np.float32] = np.zeros_like(self.distances)

    # read-only
    @property
    def attractiveness_matrix(self):
        # define getter to prevent direct value assignments
        # it should only be updated by method `update_attractiveness`
        return self.__attractiveness_matrix
    
    # read-only
    @property
    def pheromone_matrix(self):
        # define getter to prevent direct value assignments
        # it should only be updated by method `update_pheromone`
        return self.__pheromone_matrix
    
    def update_attractiveness(self, from_node: int, to_node: int, new_pheromone: float) -> None:
        self.__attractiveness_matrix


    def __initialize_attractiveness_matrix(self) -> NDArray[np.float32]:
        attractiveness_matrix: NDArray[np.float32] = np.zeros_like(self.distances)
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                if self.is_feasbile_transition(from_node=i, to_node=j):
                    attractiveness_matrix[i, j] = AntSystem.attractiveness_func(distance=self.distances[i, j])
        
        return attractiveness_matrix
    
    @staticmethod
    def attractiveness_func(distance: float) -> float:
        return 1 / (distance + 1e-10)
    
    @staticmethod
    def pheromone_func(distance: float) -> float:
        return 1 / (distance + 1e-10)
    
    def find_allowed_nodes(self, traveled_route: List[int]) -> Set[int]:
        at_node = traveled_route[-1]
        return {
            node 
            for node in range(self.n + 2) 
            if node not in traveled_route 
                and self.is_feasbile_transition(from_node=at_node, to_node=node)
        }
    
    def compute_probabilities(self, traveled_route: List[int]) -> Tuple[List[int], List[float]]:
        at_node: int = traveled_route[-1]
        allowed_nodes: List[int] = list(self.find_allowed_nodes(traveled_route=traveled_route))
        pheromones: NDArray[np.float32] = self.pheromone_matrix[at_node, allowed_nodes]
        attractivenesses: NDArray[np.float32] = self.attractiveness_matrix[at_node, allowed_nodes]
        numerators: NDArray[np.float32] = (attractivenesses ** self.alpha) * (pheromones ** self.beta)
        denominator: float = numerators.sum()
        probabilities: List[float] = list(numerators / denominator)
        return allowed_nodes, probabilities
    
    @staticmethod
    def select_next_node(allowed_nodes: List[int], probabilities: List[float]) -> int:
        return random.choices(population=allowed_nodes, weights=probabilities, k=1).pop()

    @track_time
    def find_route(self) -> Route:
        timesteps: int = self.n + 1
        deposited_pheromones: NDArray[np.float32] = np.zeros(
            shape=(timesteps, self.n_ants, self.n + 2, self.n + 2),
            dtype=np.float32,
        )
        for timestep in range(timesteps):
            for k in range(self.n_ants):
                traveled_route: List[int] = [0]
                at_node: int = traveled_route[-1]
                allowed_nodes, probabilities = self.compute_probabilities(traveled_route=traveled_route)
                next_node: int = AntSystem.select_next_node(
                    allowed_nodes=allowed_nodes, 
                    probabilities=probabilities
                )
                deposited_pheromone: float = self.pheromone_func(distance=self.distances[at_node, next_node])
                deposited_pheromones[timestep, k, at_node, next_node] = deposited_pheromone

                # update


