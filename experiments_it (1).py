"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import numpy as np
from poisson_problem_2d import plot_error, plot_surface_comparison, compute_iter_error, plot_error_k
from poisson_problem_2d import plot_error_both, plot_error_text, plot_error_text_1n, plot_surface_dif
from time_comp import benchmark, plot_benchmark
from block_matrix_2d import BlockMatrix
from poisson_problem_2d import rhs
from linear_solvers import solve_lu, solve_cg

pi = np.pi
sin = np.sin
cos = np.cos

def exact_u_func(x, kappa=1):
    """
    The exact solution of the Poisson problem:
    u(x) = x1 * x2 * sin(kappa * pi * x1) * sin(kappa * pi * x2)
    Parameters
    ----------
    x : float array of shape (2,)
        Point at which the exact solution is evaluated

    Returns
    -------
    float
        The value of the exact solution at point x
    """
    return x[0] * x[1] * sin(kappa * pi * x[0]) * sin(kappa * pi * x[1])

def f_source(x, kappa=1):
    """
    Right-hand side function f corresponding to the exact solution u.
    Computes f(x) = -Delta u(x) where u is given by exact_u_func

    Parameters
    ----------
    x : float array of shape (2,)
        Point at which the exact solution is evaluated

    Returns
    -------
    float
        The value of the right-hand side function f at point x
    """
    k = kappa * pi

    # Term bezüglich x1
    # u = (x1*sin(k*x1)) * (x2*sin(k*x2))
    # d^2/dx1^2 = (2*k*cos(k*x1) - x1*k^2*sin(k*x1)) * (x2*sin(k*x2))
    term1 = (2*k*cos(k*x[0]) - x[0]*(k**2)*sin(k*x[0])) * (x[1]*sin(k*x[1]))

    # Term bezüglich x2
    term2 = (x[0]*sin(k*x[0])) * (2*k*cos(k*x[1]) - x[1]*(k**2)*sin(k*x[1]))

    laplace = term1 + term2
    return -laplace

def cg(n):
    """
    Solve the linear system arising from a 2D Poisson problem using the 
    Conjugate Gradient (CG) method.

    Constructs the block matrix representation of the discretized Laplacian 
    and applies the CG solver to compute the solution.

    Parameters
    ----------
    n : int
        The number of grid points in one spatial dimension (grid size is n x n).

    Returns
    -------
    None
        The function directly calls `solve_cg` and does not return a value.
    """
    bm = BlockMatrix(n)
    # pylint: disable=invalid-name
    A = bm.get_sparse()
    solve_cg(A, rhs(n, f_source) / n**2, np.array([0 for i in range((n-1)**2)]),
             params={"eps": (1 / n)**4, "max_iter": 10000000000000, "var_x": 1e-16})

def lu(n):
    """
    Solve the linear system arising from a 2D Poisson problem using LU decomposition.

    Constructs the block matrix representation of the discretized Laplacian, 
    performs LU factorization, and solves the system using the factors.

    Parameters
    ----------
    n : int
        The number of grid points in one spatial dimension (grid size is n x n).

    Returns
    -------
    None
        The function directly calls `solve_lu` and does not return a value.
    """
    bm = BlockMatrix(n)
    # pylint: disable=invalid-name
    P, L, U = bm.get_lu()
    solve_lu(P, L, U, rhs(n, f_source) / n**2)

#pylint: disable=missing-function-docstring
def main():
    print("Wähle ein kappa")
    k = float(input("kappa = ").strip())

    print("1 — Funktion u und die approximierte Lösung mit cg "
    "(plot_surface_comparison)")
    print("2 — Approximationsfehler für verschiedene N mit cg "
    "(plot_error)")
    print("3 — Approximationsfehler für verschiedene N und unterschiedliche k mit cg "
    "(plot_error)")
    print("4 — Approximationsfehler der Iteration "
    "(compute_iter_error)")
    print("5 — Approximationsfehler unterschiedlicher k "
    "(plot_error_k)")
    print("6 — Approximationsfehler LU, cg "
    "(plot_error_both)")
    print("7 — Approximationsfehler mit cg in Text "
    "(plot_error_text)")
    print("8 — Approximationsfehler für verschiedene k mit cg in Text "
    "(plot_error_text)")
    print("9 — Approximationsfehler für verschiedene k_var mit cg in Text "
    "(plot_error_text)")
    print("10 — Approximationsfehler für nur ein n und verschiedene k_var mit cg in Text "
    "(plot_error_text_1n)")
    print("11 — Differenz von Funktion u und der approximierte Lösung mit cg "
    "(plot_surface_dif)")
    print("12 — Laufzeitvergleich von LU-Zerlegung und cg "
    "(benchmark)")
    print("13 — Laufzeitvergleich für verschiedene n von LU-Zerlegung und cg "
    "(plot_benchmark)")

    print("0 — Beenden")

    choice = input("Your choice: ").strip()

    if choice == "1":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_surface_comparison(n, exact_u_func, f_source, kappa=k, method='cg')
    elif choice == "2":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_error(n, exact_u_func, f_source, kappa=k, method='cg')
    elif choice == "3":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        print("\nFür welches k möchten Sie den Plot sehen?")
        k0 = int(input("k = ").strip())
        plot_error(n, exact_u_func, f_source, kappa=k, method='cg', k=k0)
    elif choice == "4":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        compute_iter_error(n, exact_u_func, f_source, k)
    elif choice == "5":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_error_k(n, exact_u_func, f_source, kappa=k)
    elif choice == "6":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_error_both(n, exact_u_func, f_source, kappa=k)
    elif choice == "7":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_error_text(n, exact_u_func, f_source, kappa=k, method='cg')
    elif choice == "8":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        print("\nFür welches k möchten Sie den Plot sehen?")
        k0 = int(input("k = ").strip())
        plot_error_text(n, exact_u_func, f_source, kappa=k, method='cg', k=k0)
    elif choice == "9":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        print("\nFür welches k möchten Sie den Plot sehen?")
        k0 = int(input("k = ").strip())
        plot_error_text(n, exact_u_func, f_source, kappa=k, method='cg', k_var=k0)
    elif choice == "10":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        print("\nFür welches k möchten Sie den Plot sehen?")
        k0 = int(input("k = ").strip())
        plot_error_text_1n(n, exact_u_func, f_source, kappa=k, method='cg', k_var=k0)
    elif choice == "11":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_surface_dif(n, exact_u_func, f_source, kappa=k, method='cg')
    elif choice == "12":
        print("\nFür welches n möchten Sie die Laufzeit vergleichen?")
        n = int(input("n = ").strip())
        print("\nWie viel Mal soll es testen?")
        runs = int(input("runs = ").strip())
        avg1, avg2 = benchmark(cg, lu, n, runs)
        print(f"Durchschnittliche Laufzeit Funktion CG: {avg1:.8f} Sekunden")
        print(f"Durchschnittliche Laufzeit Funktion LU: {avg2:.8f} Sekunden")
    elif choice == "13":
        print("\nFür welches n möchten Sie die Laufzeit vergleichen?")
        n = int(input("n = ").strip())
        print("\nWie viel Mal soll es testen?")
        runs = int(input("runs = ").strip())
        plot_benchmark(n, cg, lu, runs)

    elif choice == "0":
        print("Exiting program.")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
