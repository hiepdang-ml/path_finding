from typing import List, Tuple, Set
import argparse
import random

from utils.type_alias import Path, Cost, Result
from utils.functional import track_time
from utils.exception import NeedDifferentCrossoverPoints

from base import Solver


class GeneticAlgorithm(Solver):

    def __init__(
        self, 
        csv_path: str, 
        population_size: int,
        n_offsprings_per_generation: int,
        mutation_rate: float,
        diversity_rate: float,
        patience: int,
    ) -> None:
        
        assert 0 <= diversity_rate <= 1
        
        super().__init__(csv_path)
        self.population_size: int = population_size
        self.n_offsprings_per_generation: int = n_offsprings_per_generation
        self.mutation_rate: float = mutation_rate
        self.diversity_rate: float = diversity_rate
        self.patience: int = patience

        # Initialization:
        self.discovered_paths: Set[Path] = self.get_random_feasible_paths(n=self.population_size)
        self.__population: List[Result] = [
            (self.compute_cost(path), path)
            for path in self.discovered_paths
        ]
    
    # read-only
    @property
    def population(self) -> List[Result]:
        # define getter to prevent direct manipulations
        # it should only be updated by method `update_population`
        return self.__population
    
    def update_population(self, offsprings: Set[Path]) -> None:
        # `offsprings` set is expected to contain new individuals only
        assert offsprings.isdisjoint(self.population)
        # `offsprings` could be in any length
        individuals: List[Result] = self.__population + [
            (self.compute_cost(path=offspring), offspring) 
            for offspring in offsprings
        ]
        individuals.sort(key=lambda x: x[0])

        # population is formed by selecting the most elite individuals and random inferior individuals
        expected_n_inferiors: int = int(self.diversity_rate * self.population_size)
        # there could be not enough inferior individuals
        actual_n_inferiors: int = min(expected_n_inferiors, len(individuals) - self.population_size)
        inferiors: List[Result] = random.sample(
            population=individuals[self.population_size:], 
            k=actual_n_inferiors
        )
        # get most elite individuals
        n_elites: int = self.population_size - actual_n_inferiors
        elites: List[Result] = individuals[:n_elites]
        self.__population = elites + inferiors
        print(f'Updated population with (#elites)-(#inferiors): {n_elites}-{actual_n_inferiors}')

    @staticmethod
    def tournament_selection(population: List[Result], tournament_size: int = 3) -> Path:
        tournament_individuals: List[Result] = random.choices(population=population, k=tournament_size)
        return min(tournament_individuals, key=lambda x: x[0])[1]

    def selection(self, population: List[Result]) -> Tuple[List[Path], List[Path]]:
        parent1s: List[Path] = [
            self.tournament_selection(population=population, tournament_size=3)
            for _ in range(self.n_offsprings_per_generation // 2)
        ]
        parent2s: List[Path] = [
            self.tournament_selection(population=population, tournament_size=3)
            for _ in range(self.n_offsprings_per_generation // 2)
        ]
        return parent1s, parent2s
    
    def custom_ox1_crossover(self, parent1: Path, parent2: Path) -> Path:
        p1: List[int] = list(parent1[1: -1])
        p2: List[int] = list(parent2[1: -1])
        assert len(p1) == self.n
        assert len(p2) == self.n
        
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

        offspring: Path = tuple([0] + offspring + [self.n + 1])
        assert None not in offspring
        assert self.is_feasible_path(offspring)
        return offspring
    
    def crossover(self, parent1s: List[Path], parent2s: List[Path]) -> List[Path]:
        offsprings: List[Path] = []
        for parent1, parent2 in zip(parent1s, parent2s):
            offspring1: Path = self.custom_ox1_crossover(parent1, parent2)
            offspring2: Path = self.custom_ox1_crossover(parent2, parent1)
            offsprings.extend([offspring1, offspring2])
        return offsprings

    def custom_right_rotation_mutation(self, offspring: Path) -> Path:
        mutated_offspring: List[int] = list(offspring)

        # selected group(s) for right rotation
        if random.random() < self.mutation_rate:
            selected_group: Set[int] = random.choice(
                seq=self.node_groups[1:],    # skip group0
            )
            indices: List[int] = [i for i in range(len(offspring)) if offspring[i] in selected_group]
            new_indices: List[int] = (indices * 2)[-1-len(indices):-1]  # right shift by 1
            for i, j in zip(indices, new_indices):
                mutated_offspring[i] = offspring[j]
        
        mutated_offspring: Path = tuple(mutated_offspring)
        assert self.is_feasible_path(mutated_offspring)
        return mutated_offspring

    def mutation(self, offsprings: List[Path]) -> List[Path]:
        mutated_offsprings: List[Path] = [
            self.custom_right_rotation_mutation(offspring)
            for offspring in offsprings
        ]
        return mutated_offsprings

    @track_time
    def find_path(self) -> Result:
        best_cost: Cost; best_path: Path
        best_cost, best_path = min(self.population, key=lambda x: x[0])

        no_improvements: int = 0
        while no_improvements < self.patience:
            print(f'Number of discovered paths: {len(self.discovered_paths)}')
            print(f'Best path found so far: {best_path}, best_cost: {best_cost}')
            parent1s: List[Path]; parent2s: List[Path]
            parent1s, parent2s = self.selection(population=self.population)
            offsprings: List[Path] = self.crossover(parent1s, parent2s)
            offsprings: List[Path] = self.mutation(offsprings)

            new_paths: Set[Path] = set(offsprings) - self.discovered_paths
            n_new_paths: int = len(new_paths)
            print(f'Number of new offsprings: {n_new_paths}')
            
            self.discovered_paths.update(new_paths)
            # only update offsprings that are never seen before to the population
            self.update_population(offsprings=new_paths)
            cost: Cost; path: Path
            cost, path = min(self.population, key=lambda x: x[0])

            if cost < best_cost:
                best_cost = cost
                best_path = path
                no_improvements = 0
            else:
                no_improvements += 1
                print(f'No improvements: {no_improvements}/{self.patience}')

            print('---------------')
        
        return best_cost, best_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm')
    parser.add_argument('--csv_path', '-f', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--population_size', '-s', type=int, default=500, help='Fixed population size in each generation')
    parser.add_argument('--n_offsprings_per_generation', '-n', type=int, default=500, help='Number of offsprings bred in each generation')
    parser.add_argument('--mutation_rate', '-m', type=float, default=0.6, help='Mutation probability of an offspring')
    parser.add_argument('--diversity_rate', '-d', type=float, default=0.2, help='Proportion of random inferiors in the population')
    parser.add_argument('--patience', '-p', type=int, default=1000, help='Maximum consecutive generations of no improvements')
    args: argparse.Namespace = parser.parse_args()

    solver: Solver = GeneticAlgorithm(**vars(args))
    r: Result = solver.find_path()
    print(f'Found solution: {r}')


if __name__ == '__main__':
    main()


