from typing import List, Tuple
import argparse
import random
import string


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
            row: List[str] = random.choices(population=string.digits, k=M, weights=weights)
            if i == s_loc[0]:
                row[s_loc[1]] = 's'
            if i == f_loc[0]:
                row[f_loc[1]] = 'f'

            self.inputs += f"{''.join(row)}\n"

    def to_file(self):
        with open(file='ocean.in', mode='w') as file:
            file.write(self.inputs)


def main():
    parser = argparse.ArgumentParser(description='Produce random input files for testing purpose')
    parser.add_argument('-N', type=int, required=True, help='Number of rows of the matrix')
    parser.add_argument('-M', type=int, required=True, help='Number of columns of the matrix')
    parser.add_argument('-s', nargs=2, type=int, required=True, help='Start location in the format x y')
    parser.add_argument('-f', nargs=2, type=int, required=True, help='Finish location in the format x y')
    parser.add_argument('-p', type=float, default=0.2, help='Probability of land in the map')
    parser.add_argument('--seed', type=int, default=42, help='Random seed ensure reproducibility')
    args: argparse.Namespace = parser.parse_args()

    factory = InputFactory(N=args.N, M=args.M, s_loc=tuple(args.s), f_loc=tuple(args.f), p_land=args.p, seed=args.seed)
    factory.to_file()


if __name__ == '__main__':
    main()


