"""Compare complete GA runs with representative fitness workloads.

Run from the repository root with PyGAD installed:
    python examples/benchmarks/parallel_processing.py --workload cpu
    python examples/benchmarks/parallel_processing.py --workload io
    python examples/benchmarks/parallel_processing.py --workload numpy

Timings include worker startup and shutdown. They are measurements for
this machine and workload, not performance guarantees or unit tests.
"""

import argparse
import functools
import statistics
import time

import numpy
import pygad


def fitness(workload, ga, solution, index):
    if workload == "cpu":
        value = 0
        for _ in range(200000):
            value = (value * 1664525 + 1013904223) & 0xffff
    elif workload == "io":
        time.sleep(0.005)
    return -float(numpy.sum(solution * solution))


def batch_fitness(ga, solutions, indices):
    return -numpy.sum(solutions * solutions, axis=1)


def benchmark(workload, samples):
    configurations = [("serial", None, None), ("threads", ["thread", 2], None),
                      ("processes", ["process", 2], None)]
    if workload == "numpy":
        configurations.append(("vectorized batch", None, 24))
    baseline = None
    for label, mode, batch_size in configurations:
        durations = []
        for _ in range(samples):
            ga = pygad.GA(num_generations=3, sol_per_pop=24,
                          num_parents_mating=4, num_genes=64,
                          fitness_func=(batch_fitness if batch_size else
                                        functools.partial(fitness, workload)),
                          fitness_batch_size=batch_size,
                          parallel_processing=mode, keep_elitism=0,
                          keep_parents=0, random_seed=42, suppress_warnings=True)
            start = time.perf_counter()
            ga.run()
            durations.append(time.perf_counter() - start)
            if baseline is None:
                baseline = ga.population.copy()
            else:
                numpy.testing.assert_allclose(ga.population, baseline)
        print(f"{label:18s} median={statistics.median(durations):.6f}s "
              f"samples={durations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("cpu", "io", "numpy"), default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    benchmark(args.workload, args.samples)
