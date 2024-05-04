from abc import ABC, abstractmethod
from typing import List, Tuple
import heapq

from solution1.solve import Solver


class Solver(ABC):

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

        self.visited: List[List[bool]]
        self._reset_visited()

    def _reset_visited(self) -> None:
        self.visited = [[False] * self.M for _ in range(self.N)]

    def unvisited_land(self, i: int, j: int) -> bool:
        return (
            0 <= i < self.N                 # valid loc
            and 0 <= j < self.M             # valid loc
            and not self.visited[i][j]      # not visited
            and self.matrix[i][j] == '0'    # is land
        )

    @abstractmethod
    def to_file(self) -> None:
        pass


class DFS(Solver):

    def find_island_size(self, i: int, j: int) -> int:
        if not self.unvisited_land(i, j):
            # reach un-land cell -> stop, do not add up the count
            return 0

        self.visited[i][j] = True
        # count the current cell and recursively count adjacent cells
        return (
            + 1
            + self.find_island_size(i+1, j) 
            + self.find_island_size(i-1, j) 
            + self.find_island_size(i, j+1) 
            + self.find_island_size(i, j-1)
        )
    
    def count_islands(self) -> Tuple[int, List[int]]:
        self._reset_visited()
        island_sizes: List[int] = []
        for i in range(self.N):
            for j in range(self.M):
                if self.unvisited_land(i, j):
                    island_sizes.append(self.find_island_size(i, j))

        island_sizes.sort()
        return len(island_sizes), island_sizes

    def to_file(self) -> None:
        n_islands, island_sizes = self.count_islands()
        
        with open(file='ocean.out1', mode='w') as file:
            file.write(
                f"{n_islands}\n"
                f"{','.join(map(str, island_sizes))}"
                f"{'.' if n_islands else ''}"
            )


class Dijkstra(Solver):

    def find_safest_path(self) -> int:
        # reset the visting record
        self._reset_visited()
        
        # priority queue for the cost from start point
        priority_queue: List[Tuple[int, Tuple[int, int]]] = [(0, self.s_loc)]
        best_cost_matrix: List[List[float|int]] = [
            [float('inf')] * self.M for _ in range(self.N)
        ]

        # set the cost of start location to 0
        start_x: int = self.s_loc[0]
        start_y: int = self.s_loc[1]
        best_cost_matrix[start_x][start_y] = 0
        
        # main loop
        directions: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
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

    def to_file(self) -> None:
        with open(file='ocean.out2', mode='w') as file:
            file.write(str(self.find_safest_path()))


def main():
    counter = DFS()
    counter.to_file()
    finder = Dijkstra()
    finder.to_file()


if __name__ == '__main__':
    main()



