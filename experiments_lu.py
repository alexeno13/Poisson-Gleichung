"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import numpy as np
from block_matrix_2d import plot_LU_A_N
from poisson_problem_2d import plot_error, plot_surface_comparison
from sparse_A_Matrix import show_N, show_n_log

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
#pylint: disable=missing-function-docstring
def main():
    print("Wähle ein k")
    k = float(input("kappa = ").strip())

    print("\nWählen Sie aus, welches Diagramm angezeigt werden soll:")
    print("1 — Funktion u und die approximierte Lösung "
    "(plot_surface_comparison)")
    print("2 — Approximationsfehler für verschiedene N "
    "(plot_error)")
    print("3 — Speicherplatzbedarf von sparse A und LU für N "
    "(plot_LU_A_N)")
    print("4 — Speicherplatzbedarf von sparse A und volle A für n "
    "(show_n_log)")
    print("5 — Speicherplatzbedarf von sparse A und volle A für N "
    "(show_N)")

    print("0 — Beenden")

    choice = input("Your choice: ").strip()

    if choice == "1":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_surface_comparison(n, exact_u_func, f_source, kappa=k)
    elif choice == "2":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_error(n, exact_u_func, f_source, kappa=k)
    elif choice == "3":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        plot_LU_A_N(n)
    elif choice == "4":
        print("\nFür welches n möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        show_n_log(n)
    elif choice == "5":
        print("\nFür welches N möchten Sie den Plot sehen?")
        n = int(input("n = ").strip())
        show_N(n)

    elif choice == "0":
        print("Exiting program.")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
