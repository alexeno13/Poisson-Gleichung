"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""
from block_matrix_2d import BlockMatrix
import scipy.sparse as sp
import scipy.linalg as lin
import numpy as np
import matplotlib.pyplot as plt


pi = np.pi
sin = np.sin
cos = np.cos

def solve_lu(p, l, u, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.

    Parameters
    ----------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """
    bp = p.T @ b
    y = []
    sum0 = 0
    # pylint: disable=consider-using-enumerate
    for i in range(len(l)):
        sum0 = 0
        for j in range(i):
            sum0 += y[j] * l[i][j]
        y.append(bp[i] - sum0)

    x = [0 for i in range(len(l))]
    sum0 = 0
    for i in range(len(l)):
        sum0 = 0
        for j in range(i):
            sum0 += x[len(l) - j -1] * u[len(l) - i - 1][len(l) - j - 1]
        x[len(l) - i -1] = (y[len(l) - i - 1] - sum0)/u[len(l) - i -1][len(l) - i -1]

    return np.array(x)

# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value
# pylint: disable=use-dict-literal
# pylint: disable=unused-argument
def solve_sor(A, b, x0,
              params=dict(eps=1e-8, max_iter=1000, var_x=1e-4),
              omega=1.5):
    """ Solves the linear system Ax = b via the successive over relaxation method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    omega : float, optional
        relaxation parameter

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """
    # pylint: disable=unnecessary-pass
    pass


# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value
# pylint: disable=use-dict-literal
# pylint: disable=unused-argument
def solve_gs(A, b, x0,
             params=dict(eps=1e-8, max_iter=1000, var_x=1e-4)):
    """ Solves the linear system Ax = b via the Jacobi method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """
    # pylint: disable=unnecessary-pass
    pass

# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value
# pylint: disable=use-dict-literal
# pylint: disable=unused-argument
def solve_es(A, b, x0, params=dict(eps=1e-8, max_iter=1000, var_x=1e-4)):
    """ Solves the linear system Ax = b via the Gauss-Seidel method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinitiy norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """
    # pylint: disable=unnecessary-pass
    pass

# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value
# pylint: disable=use-dict-literal
# pylint: disable=unused-argument
def solve_cg(A, b, x0,
             params=dict(eps=1e-11, max_iter=100000000000000, var_x=1e-16)):
    """ Solves the linear system Ax = b via the conjugated gradient method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """

    if params['eps'] <= 0 :
        raise ValueError("eps <= 0")
    if params['max_iter'] <= 0 :
        raise ValueError("max_iter <= 0")
    if params['var_x'] <= 0 :
        raise ValueError("var_x <= 0")


    k = 0
    r = b - A @ x0
    d = r
    stop = True
    r_list = [r]
    x_list = [x0]
    x = x0

    while stop:
        z = A @ d
        alpha = (r @ r) / (z @ d)
        r_last = r
        r = r - alpha * z
        #print('r:' + str(r))

        beta = (r @ r)/(r_last @ r_last)
        x = x + alpha * d
        #print('x:' + str(x))
        d = r + beta * d
        k += 1
        r_list.append(r)
        x_list.append(x)
        if (np.linalg.norm(r, ord=np.inf) <= params['eps'] * np.linalg.norm(r_list[0], ord=np.inf)):
            key = 'eps'
            stop = False
            print('eps')
        if (k >= params['max_iter']):
            key = 'max_iter'
            stop = False
            print('max_iter')
        if (np.linalg.norm(x - x_list[-2], ord=np.inf) < params['var_x']):
            key = 'var_x'
            stop = False
            print('var_x')
        if (k >= len(b)):
            key = 'max_len_n'
            stop = False
            print('max_len_n')

    return key, x_list, r_list
