"""
Author: Alexander Huhn, Fedir Deineko
Date: 12.12.2025
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as lin
import matplotlib.pyplot as plt

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    n : int
        Number of intervals in each dimension.

    Raises
    ------
    ValueError
        If n < 2.
    """

    def __init__(self, n):
        self.n = n
        # pylint: disable=invalid-name
        self.N = (n - 1) ** 2
        # pylint: disable=invalid-name
        self.A_sparse = None
        # pylint: disable=invalid-name
        self.L = None
        # pylint: disable=invalid-name
        self.U = None
        # pylint: disable=invalid-name
        self.P = None
        if n < 2:
            raise ValueError("n must be at least 2.")

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """

        if self.A_sparse is not None:
            return self.A_sparse
        matrix_c = sp.diags([-1, 4, -1], offsets=[-1, 0, 1],shape=(self.n-1,self.n-1), format='csr')
        # pylint: disable=consider-using-generator
        cs = tuple([matrix_c for i in range(self.n -1)])
        matrix_a = sp.block_diag(cs)
        offset=[-(self.n - 1), (self.n - 1)]
        matrix_i = sp.diags([-1,-1],offsets=offset,shape=((self.n-1)**2,(self.n-1)**2),format='csr')
        self.A_sparse = matrix_a + matrix_i
        return self.A_sparse 

    def eval_sparsity(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        matrix_a = self.get_sparse()
        nnz = matrix_a.nnz
        total_elements = self.N * self.N
        rel_nnz = nnz / total_elements
        return nnz, rel_nnz

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            upper triangular matrix of LU-decomposition
        """
        if self.P is not None and self.L is not None and self.U is not None:
            return self.P, self.L, self.U
        if self.A_sparse is None:
            self.get_sparse()
        # pylint: disable=unbalanced-tuple-unpacking
        self.P, self.L, self.U = lin.lu(self.A_sparse.toarray())
        return self.P, self.L, self.U

    def eval_sparsity_lu(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        p, l, u = self.get_lu()
        nnz = np.count_nonzero(l) + np.count_nonzero(u) - (self.n - 1) ** 2
        total_elements = self.N * self.N
        rel_nnz = nnz / total_elements
        return nnz, rel_nnz

    def show_nonzeros(self):
        """ Plots the sparsity pattern of the represented matrix.
        """
        matrix_a = self.get_sparse()

        plt.spy(matrix_a, markersize=1)
        plt.title("Sparsity pattern of the block matrix")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def get_cond(self):
        """ Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to the infinity-norm
        """
        matrix_a = self.get_sparse()
        return np.linalg.cond(matrix_a.toarray(), np.inf)

# pylint: disable=invalid-name
def plot_LU_A_N(n):
    """ Plots the sparsity pattern of the block matrix for given n.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    """
    x = np.array([(i-1)**2 for i in range(2, n+1)])
    lu_sparsity_list = []
    a_sparsity_list = []

    ns = [2]
    hg = 10
    for i in range(2, n+1):
        if (i-1)**2 >= hg:
            ns.append(i)
            hg *= 10

    for i in ns:
        bm = BlockMatrix(i)
        lu_sparsity_list.append(bm.eval_sparsity_lu()[0])
        a_sparsity_list.append(bm.eval_sparsity()[0])

    ns = [(i-1)**2 for i in ns]
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(ns, a_sparsity_list, 'k-', label="Anzahl von Einträgen in sparse Matrix A")
    plt.plot(ns, lu_sparsity_list, 'b-', label="Anzahl von Einträgen in LU-Zerlegung")
    plt.plot(x, [3 * i**1.5 for i in x], 'm--', label="Ordnung $O(N^{1.5})$")
    plt.plot(x, [ i**2 for i in x], 'r--', label="Ordnung $O(N^2)$")
    plt.plot(x, 7*x, 'g--', label="Ordnung $O(N)$")
    plt.title("Speicherplatzbedarf von sparse A und LU für N")
    plt.xlabel('N')
    plt.ylabel('Speicherplatz')
    plt.legend()
    plt.grid(True)
    plt.show()

def rhs(n, f):
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
            b.append(f(np.array([j/n, k/n])) / n**2)
    return np.array(b)
if __name__ == "__main__":
    np.set_printoptions(precision=1, suppress=True)
    bm_test = BlockMatrix(4)
    print(bm_test.get_lu())
