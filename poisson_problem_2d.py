"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import numpy as np
import matplotlib.pyplot as plt
from block_matrix_2d import BlockMatrix
from linear_solvers import solve_lu, solve_cg

pi = np.pi
sin = np.sin
cos = np.cos

def rhs(n, f, kappa=1):
    """ Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If n < 2.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    b = []
    for k in range(1,n):
        for j in range(1,n):
            b.append(f(np.array([j/n, k/n]), kappa))
    return np.array(b)



def idx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    nx0, nx1 = nx
    return (nx0 - 1) * (n - 1) + nx1


def inv_idx(m, n):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.
    
    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """

    nx0 = m // (n - 1) + 1
    nx1 = m % (n - 1)
    if nx1 == 0:
        nx0 -= 1
        nx1 = n - 1
    return [nx0, nx1]


def compute_error(n, hat_u, u, kappa=1):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """
    max_error = 0
    for m in range((n - 1)**2):
        nx = inv_idx(m + 1, n)
        x = np.array([nx[0] / n, nx[1] / n])
        if u(x, kappa) == 0:
            print(u(x, kappa))
        error = abs(hat_u[m] - u(x, kappa))
        #pylint: disable=consider-using-max-builtin
        if error > max_error:
            max_error = error
    return max_error


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def plot_error(n, u, f, logscale=True, kappa=1, method='LU', k=None):
    """ Plots the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm for different N.
    Parameters
    ----------
    n : int
        Maximum number of intervals in each dimension.
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    logscale : bool, optional
        If True, the plot is shown in log-log scale. Default is True.
    kappa : float, optional
        Parameter kappa in the definition of the exact solution and
        right-hand side function. Default is 1.
    """
    x = np.array([(i-1)**2 for i in range(2, n+1)])
    error_list = []
    condition_list = []
    ns = [3]
    hg = 10
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in range(len(ns) - 1):
        ns.append(int(np.ceil((ns[i] + ns[i+1])/2)))

    ns = sorted(list(set(ns)))

    for i in ns:
        u_hat = compute_u(i, f, kappa, method, k)
        error_list.append(compute_error(i, u_hat, u, kappa))
        condition_list.append(BlockMatrix(i).get_cond())

    xs = np.array([(i-1)**2 for i in ns])

    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.plot(xs, error_list, 'b-x', label="Fehler von $u$")

    plt.plot(xs, condition_list, 'r-o', label="Konditionszahl von $A$")
    plt.plot(x, 1.05 * x, 'm--', label="Ordnung $O(N)$")
    plt.plot(x, [1/(2*(xi**2)) for xi in range(2, n+1)], 'k--', label="Ordnung $O(h^2)$")
    plt.plot(x, [1/(2*xi) for xi in x], 'g--', label="Ordnung $O(N^{-1})$")
    plt.title(rf"Fehler von $u$ abhängig von $N$ für kappa={kappa}")
    plt.xlabel('N')
    plt.ylabel('Fehler')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_text(n, u, f, kappa=1, method='LU', k=None, k_var=None):
    """ Plots the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm for different N.
    Parameters
    ----------
    n : int
        Maximum number of intervals in each dimension.
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    logscale : bool, optional
        If True, the plot is shown in log-log scale. Default is True.
    kappa : float, optional
        Parameter kappa in the definition of the exact solution and
        right-hand side function. Default is 1.
    """

    error_list = []
    ns = [3]
    hg = 10
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in range(len(ns) - 1):
        ns.append(int(np.ceil((ns[i] + ns[i+1])/2)))

    ns = sorted(list(set(ns)))

    for i in ns:
        u_hat = compute_u(i, f, kappa, method, k, k_var)
        error_list.append(compute_error(i, u_hat, u, kappa))


    xs = np.array([(i-1)**2 for i in ns])

    for i in zip(xs, error_list):
        print(i)

def plot_error_text_1n(n, u, f, kappa=1, method='LU', k=None, k_var=None):
    """ Plots the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm for different N.
    Parameters
    ----------
    n : int
        Maximum number of intervals in each dimension.
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    logscale : bool, optional
        If True, the plot is shown in log-log scale. Default is True.
    kappa : float, optional
        Parameter kappa in the definition of the exact solution and
        right-hand side function. Default is 1.
    """


    u_hat = compute_u(n, f, kappa, method, k, k_var)
    error = compute_error(n, u_hat, u, kappa)

    print(str((n-1)**2))
    print(error)

def compute_u(n, f, kappa=1, method='LU', k=None, k_var=None):
    """ Computes the finite difference approximation of the solution of the
    Poisson problem at the discretization points.   
    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The
        return value is a scalar.
    kappa : float, optional
        Parameter kappa in the definition of the right-hand side function. Default is 1.
    Returns
    -------
    numpy.ndarray
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points.
    """
    bm = BlockMatrix(n)
    # pylint: disable=invalid-name
    A = bm.get_sparse()
    if method == 'LU':
        # pylint: disable=invalid-name
        P, L, U = bm.get_lu()
        u = solve_lu(P, L, U, rhs(n, f, kappa) / n**2)
        return u
    if method == 'cg':
        if (k is not None) and (k_var is None):
            param = {"eps": (1 / n)**k, "max_iter": 10000000000000, "var_x": 1e-16}
            return solve_cg(A, rhs(n, f, kappa) / n**2, np.array([0 for i in range((n-1)**2)]),
                            params=param)[1][-1]
        if (k is None) and (k_var is not None):
            param = {"eps": 1e-11, "max_iter": 10000000000000, "var_x": (1/n)**k_var}
            return solve_cg(A, rhs(n, f, kappa) / n**2,
                            np.array([0 for i in range((n-1)**2)]),
                            params=param)[1][-1]
        if (k is not None) and (k_var is not None):
            param = {"eps": (1/n)**k, "max_iter": 10000000000000, "var_x": (1/n)**k_var}
            return solve_cg(A, rhs(n, f, kappa) / n**2, np.array([0 for i in range((n-1)**2)]),
                            params=param)[1][-1]
        return solve_cg(A, rhs(n, f, kappa) / n**2,
                        np.array([0 for i in range((n-1)**2)]))[1][-1]

    raise ValueError("Invalid method. Use 'LU' or 'cg'.")

def plot_surface_comparison(n, u_func, f, kappa=1, method='LU', k=None):
    """
    Stellt approximierte und exakte Lösung grafisch gegenüber
    Parameters
    ----------
    n : int
        Maximum number of intervals in each dimension.
    u_func : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    kappa : float, optional
        Parameter kappa in the definition of the exact solution and
        right-hand side function. Default is 1.
    """
    hat_u = compute_u(n, f, kappa, method, k)

    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    # pylint: disable=invalid-name
    X, Y = np.meshgrid(x, y)
    x1 = np.linspace(0, 1, 51)
    y1 = np.linspace(0, 1, 51)
    # pylint: disable=invalid-name
    X1, Y1 = np.meshgrid(x1, y1)
    # pylint: disable=invalid-name
    U_approx = np.zeros_like(X)
    # pylint: disable=invalid-name
    U_exact = np.zeros_like(X1)

    for j_idx in range(n-1):
        for i_idx in range(n-1):
            m = idx([i_idx + 1, j_idx + 1], n)

            U_approx[j_idx+1, i_idx+1] = hat_u[m-1]

            xi = (i_idx+1)/n
            yj = (j_idx+1)/n
            #U_exact[j_idx+1, i_idx+1] = u_func([xi, yj], kappa)

    for j_idx in range(49):
        for i_idx in range(49):
            m = idx([i_idx + 1, j_idx + 1], 50)

            #U_approx[j_idx+1, i_idx+1] = hat_u[m-1]

            xi = (i_idx+1)/50
            yj = (j_idx+1)/50
            U_exact[j_idx+1, i_idx+1] = u_func([xi, yj], kappa)

    fig = plt.figure(figsize=(12, 5))

    print(kappa)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_approx, cmap='viridis', edgecolor='none')

    ax1.set_title(rf'Approximation $\hat{{u}}$ (n={n})')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X1, Y1, U_exact, cmap='plasma', edgecolor='none')
    ax2.set_title(f'Exakte Lösung $u$ (n={50})')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

def compute_iter_error(n, u, f, kappa=1):
    """
    Compute and plot the error of the iterative Conjugate Gradient (CG) solution 
    for a 2D Poisson problem as a function of iteration number.

    Evaluates the exact solution on a uniform grid, solves the system with different 
    CG parameters, and plots the convergence of the error over selected iterations.

    Parameters
    ----------
    n : int
        Number of grid points in one spatial dimension (grid size is n x n).
    u : callable
        Exact solution function u(x, kappa) for computing the reference solution.
    f : callable
        Right-hand side function f(x, kappa) corresponding to u.
    kappa : float, optional
        Parameter in the exact solution and source term, default is 1.

    Returns
    -------
    None
        Produces a plot of the error versus iteration number.
    """
    bm = BlockMatrix(n)
    # pylint: disable=invalid-name
    A = bm.get_sparse()
    u_hat = solve_cg(A, rhs(n, f, kappa) / n**2, np.array([0 for i in range((n-1)**2)]))
    u_hat_2 = solve_cg(A, rhs(n, f, kappa) / n**2, np.array([0 for i in range((n-1)**2)]),
                       params={"eps": (1/n)**4, "max_iter": 10000000000000, "var_x": 1e-16})
    error_list = []
    ns = [3]
    hg = 3
    for i in range(2, len(u_hat[1])+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 2

    ns = sorted(list(set(ns)))
    print(ns)

    for i in ns:
        error_list.append(compute_error(n, u_hat[1][i], u, kappa))

    error_list2 = []
    ns2 = [3]
    hg = 3
    for i in range(2, len(u_hat_2[1])+2):
        if (i-1)**2 >= hg:
            ns2.append(i)
            hg *= 2

    ns2 = sorted(list(set(ns2)))
    print(ns2)
    for i in ns2:
        error_list2.append(compute_error(n, u_hat_2[1][i], u, kappa))

    plt.plot(list(range(1, len(error_list) + 1)), error_list, 'b-x')
    plt.plot(list(range(1, len(error_list2) + 1)), error_list2, 'k-x')
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('Fehler')
    plt.title(rf"Fehler von $u$ abhängig von der Iteration für n={n} und kappa={kappa}")
    plt.grid(True)
    plt.show()

def plot_error_k(n, u, f, logscale=True, kappa=1):
    """
    Plot the error of the numerical solution of a 2D Poisson problem 
    as a function of grid size N for different CG solver parameters.

    Computes solutions using CG with varying tolerance settings and plots 
    the error versus N. Optionally uses logarithmic scales.

    Parameters
    ----------
    n : int
        Maximum number of grid points in one spatial dimension.
    u : callable
        Exact solution function u(x, kappa) for reference.
    f : callable
        Right-hand side function f(x, kappa) corresponding to u.
    logscale : bool, optional
        If True, both axes are plotted in log scale, default is True.
    kappa : float, optional
        Parameter in the exact solution and source term, default is 1.

    Returns
    -------
    None
        Produces a plot of the error versus N.
    """

    x = np.array([(i-1)**2 for i in range(2, n+1)])
    error_list = []
    ns = [3]
    hg = 10
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in range(len(ns) - 1):
        ns.append(int(np.ceil((ns[i] + ns[i+1])/2)))

    ns = sorted(list(set(ns)))
    for k in [-2, 0, 2, 4, 6]:
        list0 = []
        for i in ns:
            u_hat = compute_u(i, f, kappa, 'cg', k)
            list0.append(compute_error(i, u_hat, u, kappa))
        error_list.append(list0)
        #print(error_list[l])

    xs = np.array([(i-1)**2 for i in ns])
    #print(xs)

    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    for k in [[-2, 'b', error_list[0]], [0, 'm', error_list[1]], [2, 'r', error_list[2]],
              [4, 'c', error_list[3]], [6, 'y', error_list[4]]]:
        plt.plot(xs, k[2], f'{k[1]}-x', label=f"Fehler von $u$ mit {k[0]}")

    plt.plot(x, [1/(2*(xi**2)) for xi in range(2, n+1)], 'k--', label="Ordnung $O(h^2)$")
    plt.plot(x, [1/(2*xi) for xi in x], 'g--', label="Ordnung $O(N^{-1})$")
    plt.title(rf"Fehler von $u$ abhängig von $N$ für kappa={kappa}")
    plt.xlabel('N')
    plt.ylabel('Fehler')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_both(n, u, f, logscale=True, kappa=1):
    """
    Compare the error of LU decomposition and CG solver for a 2D Poisson problem 
    as a function of grid size N.

    Computes solutions using LU and CG (with different eps parameters), then plots 
    the error versus N, optionally using logarithmic scales.

    Parameters
    ----------
    n : int
        Maximum number of grid points in one spatial dimension.
    u : callable
        Exact solution function u(x, kappa) for reference.
    f : callable
        Right-hand side function f(x, kappa) corresponding to u.
    logscale : bool, optional
        If True, both axes are plotted in log scale, default is True.
    kappa : float, optional
        Parameter in the exact solution and source term, default is 1.

    Returns
    -------
    None
        Produces a plot comparing errors of LU and CG solutions.
    """
    x = np.array([(i-1)**2 for i in range(2, n+1)])
    error_list = []
    error_list0 = []
    error_list1 = []
    ns = [3]
    hg = 10
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in range(len(ns) - 1):
        ns.append(int(np.ceil((ns[i] + ns[i+1])/2)))

    ns = sorted(list(set(ns)))

    for i in ns:
        u_hat = compute_u(i, f, kappa, method='LU')
        error_list.append(compute_error(i, u_hat, u, kappa))
        u_hat0 = compute_u(i, f, kappa, method='cg')
        error_list0.append(compute_error(i, u_hat0, u, kappa))
        u_hat1 = compute_u(i, f, kappa, method='cg', k=2)
        error_list1.append(compute_error(i, u_hat1, u, kappa))

    xs = np.array([(i-1)**2 for i in ns])

    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.plot(xs, error_list, 'b-x', label="Fehler von $u$ mit LU")
    plt.plot(xs, error_list0, 'r-x', label="Fehler von $u$ mit cg und fest eps")
    plt.plot(xs, error_list1, 'm-x', label="Fehler von $u$ mit cg und eps=-2")
    plt.plot(x, [1/(2*(xi**2)) for xi in range(2, n+1)], 'k--', label="Ordnung $O(h^2)$")
    plt.plot(x, [1/(2*xi) for xi in x], 'g--', label="Ordnung $O(N^{-1})$")
    plt.title(rf"Fehler von $u$ abhängig von $N$ für kappa={kappa}")
    plt.xlabel('N')
    plt.ylabel('Fehler')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_k_var(n, u, f, logscale=True, kappa=1):
    """
    Plot the error of the numerical solution of a 2D Poisson problem 
    as a function of grid size N for CG solver with variable parameters.

    Computes solutions using CG with different variable parameters `k_var`, then plots 
    the error versus N. Optionally uses logarithmic axes.

    Parameters
    ----------
    n : int
        Maximum number of grid points in one spatial dimension.
    u : callable
        Exact solution function u(x, kappa) for reference.
    f : callable
        Right-hand side function f(x, kappa) corresponding to u.
    logscale : bool, optional
        If True, both axes are plotted in log scale, default is True.
    kappa : float, optional
        Parameter in the exact solution and source term, default is 1.

    Returns
    -------
    None
        Produces a plot of the error versus N for different variable solver parameters.
    """

    x = np.array([(i-1)**2 for i in range(2, n+1)])
    error_list = []
    ns = [3]
    hg = 10
    for i in range(2, n+2):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in range(len(ns) - 1):
        ns.append(int(np.ceil((ns[i] + ns[i+1])/2)))

    ns = sorted(list(set(ns)))
    for k in [-2, 0, 2, 4, 6]:
        list0 = []
        for i in ns:
            u_hat = compute_u(i, f, kappa, 'cg', k_var=k)
            list0.append(compute_error(i, u_hat, u, kappa))
        error_list.append(list0)
        #print(error_list[l])

    xs = np.array([(i-1)**2 for i in ns])
    #print(xs)

    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    for k in [[-2, 'b', error_list[0]], [0, 'm', error_list[1]], [2, 'r', error_list[2]],
              [4, 'c', error_list[3]], [6, 'y', error_list[4]]]:
        plt.plot(xs, k[2], f'{k[1]}-x', label=f"Fehler von $u$ mit {k[0]}")

    plt.plot(x, [1/(2*(xi**2)) for xi in range(2, n+1)], 'k--', label="Ordnung $O(h^2)$")
    plt.plot(x, [1/(2*xi) for xi in x], 'g--', label="Ordnung $O(N^{-1})$")
    plt.title(rf"Fehler von $u$ abhängig von $N$ für kappa={kappa}")
    plt.xlabel('N')
    plt.ylabel('Fehler')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_surface_dif(n, u_func, f, kappa=1, method='LU', k=None):
    """
    Plots the difference between the numerical solution and
    the exact solution of the Poisson problem
    Parameters
    ----------
    n : int
        Maximum number of intervals in each dimension.
    u_func : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    kappa : float, optional
        Parameter kappa in the definition of the exact solution and
        right-hand side function. Default is 1.
    """
    hat_u = compute_u(n, f, kappa, method, k)

    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    # pylint: disable=invalid-name
    X, Y = np.meshgrid(x, y)
    # pylint: disable=invalid-name
    U_diff = np.zeros_like(X)


    for j_idx in range(n-1):
        for i_idx in range(n-1):
            m = idx([i_idx + 1, j_idx + 1], n)

            xi = (i_idx+1)/n
            yj = (j_idx+1)/n

            U_diff[j_idx+1, i_idx+1] = hat_u[m-1] - u_func([xi, yj], kappa)


    fig = plt.figure(figsize=(8, 6))

    print(kappa)

    ax1 = fig.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_diff, cmap='viridis', edgecolor='none')

    ax1.set_title(rf'Approximation $\hat{{u}}$ (n={n})')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
