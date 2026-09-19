"""Вещественный ГА: собственные селекция, скрещивание и мутация."""
import numpy as np
from common import tournament


def objective(x):
    """Поддерживает один вектор и пакет векторов; минимум 0 при x=0."""
    x = np.asarray(x)
    return x[..., 0] ** 2 + 1_000_000 * np.sum(x[..., 1:] ** 2, axis=-1)


def reflect(x, lo=-100., hi=100.):
    """Многократное отражение вместо накопления особей на границе."""
    width = hi - lo
    return lo + width - np.abs((x - lo) % (2 * width) - width)


def solve(config, seed, scale=1., random_search=False):
    rng = np.random.default_rng(seed)
    n, d, generations = config['population'], config['dimension'], config['generations']
    population = rng.uniform(-100, 100, (n, d))
    scores = objective(population)
    best = population[np.argmin(scores)].copy()
    value = float(scores.min())
    trace = [value]
    for generation in range(generations):
        if random_search:
            children = rng.uniform(-100, 100, (n, d))
        else:
            a = population[tournament(rng, scores, n, config['tournament'])]
            b = population[tournament(rng, scores, n, config['tournament'])]
            # Выпуклая арифметическая рекомбинация, отдельный вес каждой координаты.
            alpha = rng.random((n, d))
            crossed = alpha * a + (1 - alpha) * b
            children = np.where(rng.random((n, 1)) < config['crossover_probability'], crossed, a).copy()
            progress = generation / max(1, generations - 1)
            sigma = scale * config['sigma_start'] * (config['sigma_end'] / config['sigma_start']) ** progress
            mask = rng.random((n, d)) < config['mutation_probability']
            children = reflect(children + mask * rng.normal(0, sigma, (n, d)))
        child_scores = objective(children)
        k = int(np.argmin(child_scores))
        if child_scores[k] < value:
            value, best = float(child_scores[k]), children[k].copy()
        if not random_search:
            # Поколенческая замена с одной элитой; все n потомков уже оценены.
            worst = int(np.argmax(child_scores))
            children[worst], child_scores[worst] = best, value
            population, scores = children, child_scores
        trace.append(value)
    return value, best, trace, n * (generations + 1)
