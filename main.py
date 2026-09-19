"""Запуск серии с одинаковым числом вычислений функции у всех методов."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import BASE, finish, save_json
from model import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'config.json')
    parser.add_argument('--runs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, default=BASE/'results')
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8'))
    if args.runs is not None: c['runs'] = args.runs
    if args.seed is not None: c['first_seed'] = args.seed
    if c['runs'] < 1 or c['population'] < 30 or c['generations'] < 1 or c['dimension'] != 9 or c['first_seed'] < 0:
        parser.error('Нужны runs>=1, population>=30, generations>=1, dimension=9, seed>=0')
    if not (0 <= c['mutation_probability'] <= 1 and 0 <= c['crossover_probability'] <= 1 and c['sigma_start'] > 0 and c['sigma_end'] > 0 and c['tournament'] >= 1):
        parser.error('Некорректные вероятности, sigma или размер турнира')
    rows, traces, solutions = [], {}, []
    for method, scale, random_search in [('GA sigma x1',1.,False),('GA sigma x3',3.,False),('Random',1.,True)]:
        traces[method] = []
        for run in range(c['runs']):
            seed = c['first_seed'] + run
            start = time.perf_counter()
            value, vector, trace, calls = solve(c, seed, scale, random_search)
            rows.append(dict(method=method,seed=seed,score=value,seconds=time.perf_counter()-start,evaluations=calls,feasible=int(np.all(np.abs(vector)<=100))))
            traces[method].append(trace)
            solutions.append(dict(method=method,seed=seed,value=value,vector=vector.tolist()))
        print(method, 'completed', flush=True)
    finish(args.output,c,rows,traces,'Best objective (minimize)',True)
    save_json(args.output/'solutions.json',solutions)

if __name__ == '__main__': main()
