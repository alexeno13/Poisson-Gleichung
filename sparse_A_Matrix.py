"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_sparse_memory(n):
    """ Computes the memory consumption of the sparse matrix A
    for a given n.
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    """
    return 5 * n**2 - 14 * n + 9

#pylint: disable=invalid-name
def compute_sparse_memory_N(N):
    """ Computes the memory consumption of the sparse matrix A
    for a given N.
    Parameters
    ----------
    N : int
        Number of intervals in each dimension.
    
    Returns
    -------
    float
        Memory consumption of the sparse matrix A
    """
    return 5 * N - 4 * np.sqrt(N)

def compute_full(n):
    """ Computes the memory consumption of the full matrix A
    for a given n.
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Returns
    -------
    float
        Memory consumption of the full matrix A
    """
    return (n-1)**4

#pylint: disable=invalid-name
def compute_full_N(N):
    """ Computes the memory consumption of the full matrix A
    for a given N.
    Parameters
    ----------
    N : int
        Number of intervals in each dimension.

    Returns
    -------
    float
        Memory consumption of the full matrix A
    """
    return N**2

def show_n(n, sparse, full):
    """ Plots the memory consumption of the sparse and full matrix A
    for a given n.
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    sparse : callable
        Function to compute the memory consumption of the sparse matrix A
    full : callable
        Function to compute the memory consumption of the full matrix A
    """
    x = np.linspace(0, n)
    plt.plot(x, sparse(x), 'k-', label="Anzahl von Einträgen in sparse Matrix A")
    plt.plot(x, full(x), 'b-', label="Anzahl von Einträgen in Matrix A")
    plt.title(f"Speicherplatzbedarf von sparse A und volle A für n = {n}")
    plt.xlabel('n')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_n_log(n, sparse = compute_sparse_memory, full = compute_full):
    """ Plots the memory consumption of the sparse and full matrix A
    for a given n in log-log scale.
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    sparse : callable
        Function to compute the memory consumption of the sparse matrix A
    full : callable
        Function to compute the memory consumption of the full matrix A
    """
    x = np.arange(2, n)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, sparse(x), 'k-', label="Anzahl von Einträgen in sparse Matrix A")
    plt.plot(x, full(x), 'b-', label="Anzahl von Einträgen in Matrix A")
    plt.plot(x, [ 9 * i**2 for i in x], 'r--', label="Ordnung $O(n^2)$")
    plt.plot(x, [ 2 * i**4 for i in x], 'g--', label="Ordnung $O(n^4)$")
    plt.title("Speicherplatzbedarf von sparse A und volle A für n")
    plt.xlabel('n')
    plt.ylabel('Speicherplatz')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_N(n, sparse = compute_sparse_memory_N, full = compute_full_N):
    """ Plots the memory consumption of the sparse and full matrix A
    for a given N in log-log scale.
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    sparse : callable
        Function to compute the memory consumption of the sparse matrix A
    full : callable
        Function to compute the memory consumption of the full matrix A
    """
    x = np.array([(i-1)**2 for i in range(2, n+1)])
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, sparse(x), 'k-', label="Anzahl von Einträgen in sparse Matrix A")
    plt.plot(x, full(x), 'b-', label="Anzahl von Einträgen in Matrix A")
    plt.plot(x, [ 2 * i**2 for i in x], 'r--', label="Ordnung $O(N^2)$")
    plt.plot(x, 9 * x, 'g--', label="Ordnung O(N)")
    plt.title("Speicherplatzbedarf von sparse A und volle A für N")
    plt.xlabel('N')
    plt.ylabel('Speicherplatz')
    plt.legend()
    plt.grid(True)
    plt.show()
