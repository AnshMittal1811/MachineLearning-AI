import numpy as np
from numpy.random import default_rng

def FireflyAlgorithm(function, dim, lb, ub, max_evals, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
    rng = default_rng(seed)
    fireflies = rng.uniform(lb, ub, (pop_size, dim))
    intensity = np.apply_along_axis(function, 1, fireflies)
    best = np.min(intensity)

    evaluations = pop_size
    new_alpha = alpha
    search_range = ub - lb

    while evaluations <= max_evals:
        new_alpha *= 0.97
        for i in range(pop_size):
            for j in range(pop_size):
                if intensity[i] >= intensity[j]:
                    r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                    beta = betamin * np.exp(-gamma * r)
                    steps = new_alpha * (rng.random(dim) - 0.5) * search_range
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                    fireflies[i] = np.clip(fireflies[i], lb, ub)
                    intensity[i] = function(fireflies[i])
                    evaluations += 1
                    best = min(intensity[i], best)
                    print("evaluations: {}, best results: {}".format(evaluations, best))
    return best
