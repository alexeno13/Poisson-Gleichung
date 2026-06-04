"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark(func1, func2, param, runs=10):
    """
    Measure and compare the average execution time of two functions over multiple runs.

    Parameters
    ----------
    func1 : callable
        First function to benchmark. Must accept a single argument `param`.
    func2 : callable
        Second function to benchmark. Must accept a single argument `param`.
    param : any
        Parameter to pass to both functions during each run.
    runs : int, optional
        Number of times each function is executed for averaging, default is 10.

    Returns
    -------
    tuple of float
        Average execution time (in seconds) of `func1` and `func2`.
    """
    def measure(func):
        start = time.perf_counter()
        for _ in range(runs):
            func(param)
        end = time.perf_counter()
        return (end - start) / runs

    avg1 = measure(func1)
    avg2 = measure(func2)
    return avg1, avg2

    #print(f"Durchschnittliche Laufzeit Funktion CG: {avg1:.8f} Sekunden")
    #print(f"Durchschnittliche Laufzeit Funktion LU: {avg2:.8f} Sekunden")

def plot_benchmark(n, func1, func2, runs=10):
    """
    Plot a runtime comparison between two functions (e.g., CG and LU solvers) 
    as a function of problem size N.

    Uses `benchmark` to measure average execution times for a sequence of grid sizes 
    and generates a log-scale plot of runtime versus N.

    Parameters
    ----------
    n : int
        Maximum problem size (number of grid points in one dimension).
    func1 : callable
        First function to benchmark (e.g., CG solver). Must accept a single integer argument.
    func2 : callable
        Second function to benchmark (e.g., LU solver). Must accept a single integer argument.
    runs : int, optional
        Number of runs per function for averaging, default is 10.

    Returns
    -------
    None
        Produces a plot of average runtime versus N for both functions.
    """
    ns = [3]
    hg = 3
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 2

    ns = sorted(list(set(ns)))
    print(ns)

    func1_time = []
    func2_time = []

    for i in ns:
        avg1, avg2 = benchmark(func1, func2, i, runs)
        func1_time.append(avg1)
        func2_time.append(avg2)

    xs = np.array([(i-1)**2 for i in ns])

    plt.plot(xs, func1_time, 'x-k', label="CG")
    plt.plot(xs, func2_time, 'x-b', label="LU")
    #plt.plot(ns, [1/(2*(xi**2)) for xi in ns], 'k--', label="Ordnung $O(h^2)$")
    plt.plot(ns, [1/(2*xi) for xi in ns], 'g--', label="Ordnung $O(N)$")
    plt.title(rf"Laufzeitvergleich von CG und LU für N={n}")
    plt.xlabel('N')
    plt.xscale('log')
    plt.ylabel('Laufzeit (Sekunden)')
    plt.legend()
    plt.grid(True)
    plt.show()
