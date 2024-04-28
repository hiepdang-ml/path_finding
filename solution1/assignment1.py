from typing import List, Tuple
import random
import heapq

class InputFactory:

    def __init__(
        self, 
        N: int, 
        M: int, 
        s_loc: Tuple[int, int], 
        f_loc: Tuple[int, int], 
        p_land: float = 0.2, 
        seed: int = 42
    ) -> None:
        
        self.N = N
        self.M = M
        self.s_loc = s_loc
        self.f_loc = f_loc
        self.p_land = p_land
        self.seed = seed

        assert 0. <= p_land <= 1.
        
        weights: List[float] = [p_land] + [(1 - p_land) / 9] * 9
        self.inputs: str = f'{N} {M}\n'
        
        random.seed(seed)
        for i in range(N):
            row: List[str] = random.choices(population='0123456789', k=M, weights=weights)
            if i == s_loc[0]:
                row[s_loc[1]] = 's'
            if i == f_loc[0]:
                row[f_loc[1]] = 'f'

            self.inputs += f"{''.join(row)}\n"

    def to_file(self):
        with open(file='ocean.in', mode='w') as file:
            file.write(self.inputs)


class Solver:

    def __init__(self) -> None:

        with open('ocean.in', 'r') as file:
            self.N, self.M = map(int, file.readline().split())
            self.matrix = [list(file.readline().strip()) for _ in range(self.N)]

        for i in range(self.N):
            for j in range(self.M):
                if self.matrix[i][j] == 's':
                    self.s_loc: Tuple[int, int] = (i, j)
                elif self.matrix[i][j] == 'f':
                    self.f_loc: Tuple[int, int] = (i, j)

        self.visited: List[List[bool]] = [[False]*self.M for _ in range(self.N)]

    def unvisited_land(self, i: int, j: int) -> bool:
        return (
            0 <= i < self.N                 # valid loc
            and 0 <= j < self.M             # valid loc
            and not self.visited[i][j]      # not visited
            and self.matrix[i][j] == '0'    # is land
        )

    def find_island_size(self, i: int, j: int) -> int:
        if not self.unvisited_land(i, j):
            return 0

        self.visited[i][j] = True
        return (
            1 
            + self.find_island_size(i+1, j) 
            + self.find_island_size(i-1, j) 
            + self.find_island_size(i, j+1) 
            + self.find_island_size(i, j-1)
        )
    
    def count_islands(self) -> Tuple[int, List[int]]:
        self.visited: List[List[bool]] = [[False]*self.M for _ in range(self.N)]
        island_sizes: List[int] = []
        for i in range(self.N):
            for j in range(self.M):
                if self.unvisited_land(i, j):
                    island_sizes.append(self.find_island_size(i, j))

        island_sizes.sort()
        return len(island_sizes), island_sizes

    def find_safest_path(self) -> int:

        self.visited: List[List[bool]] = [[False]*self.M for _ in range(self.N)]

        # Priority queue for the cost from start point
        priority_queue: List[Tuple[int, Tuple[int, int]]] = [(0, self.s_loc)]
        best_cost_matrix: List[List[float|int]] = [
            [float('inf')] * self.M for _ in range(self.N)
        ]
        best_cost_matrix[self.s_loc[0]][self.s_loc[1]] = 0

        directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while priority_queue:

            cost, (x, y) = heapq.heappop(priority_queue)

            if (x, y) == self.f_loc:
                return cost
            if self.visited[x][y]:
                continue

            self.visited[x][y] = True

            for dx, dy in directions:
                new_x: int = x + dx
                new_y: int = y + dy
                if (
                    0 <= new_x < self.N and 0 <= new_y < self.M     # valid loc
                    and self.matrix[new_x][new_y] != '0'            # not land
                ):
                    if self.matrix[new_x][new_y].isdigit():
                        new_cost: int = cost + int(self.matrix[new_x][new_y])
                    else:
                        new_cost: int = cost

                    if new_cost < best_cost_matrix[new_x][new_y]:
                        best_cost_matrix[new_x][new_y] = new_cost
                        heapq.heappush(priority_queue, (new_cost, (new_x, new_y)))

        return -1

    def to_files(self) -> None:
        n_islands: int
        island_sizes: List[int]
        n_islands, island_sizes = self.count_islands()

        with open(file='ocean.out1', mode='w') as file:
            file.write(
                f"{n_islands}\n"
                f"{','.join(map(str, island_sizes))}"
                f"{'.' if n_islands else ''}"
            )
        with open(file='ocean.out2', mode='w') as file:
            file.write(str(self.find_safest_path()))



if __name__ == '__main__':
    
    # input_factory = InputFactory(N=10, M=10, s_loc=(0, 2), f_loc=(9, 8), p_land=0.3, seed=42)
    # input_factory.to_file()

    solver = Solver()
    solver.to_files()








