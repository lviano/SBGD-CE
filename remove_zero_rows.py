"""
Routines for removing redundant (linearly dependent) equations from linear
programming equality constraints.
"""
# Author: Matt Haberland
# Modified version by Luca Viano

import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm


def _row_count(A):
    """
    Counts the number of nonzeros in each row of input array A.
    Nonzeros are defined as any element with absolute value greater than
    tol = 1e-13. This value should probably be an input to the function.
    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    Returns
    -------
    rowcount : 1-D array
        Number of nonzeros in each row of A
    """
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()


def _get_densest(A, eligibleRows):
    """
    Returns the index of the densest row of A. Ignores rows that are not
    eligible for consideration.
    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    eligibleRows : 1-D logical array
        Values indicate whether the corresponding row of A is eligible
        to be considered
    Returns
    -------
    i_densest : int
        Index of the densest row in A eligible for consideration
    """
    rowCounts = _row_count(A)
    return np.argmax(rowCounts * eligibleRows)


def _remove_zero_rows(A):
    """
    Eliminates trivial equations from system of equations defined by Ax = b
   and identifies trivial infeasibilities
    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    """
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    return A


def bg_update_dense(plu, perm_r, v, j):
    LU, p = plu

    vperm = v[perm_r]
    u = dtrsm(1, LU, vperm, lower=1, diag=1)
    LU[:j+1, j] = u[:j+1]
    l = u[j+1:]
    piv = LU[j, j]
    LU[j+1:, j] += (l/piv)
    return LU, p


def my_remove_redundancy_pivot_dense(A, true_rank=None):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.
    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    Returns
    -------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    keep : 1-D array
        Indices of the kept rows.
    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    """
    tolapiv = 1e-8
    tolprimal = 1e-8
    status = 0
    message = ""
    inconsistent = ("There is a linear combination of rows of A_eq that "
                    "results in zero, suggesting a redundant constraint. "
                    "However the same linear combination of b_eq is "
                    "nonzero, suggesting that the constraints conflict "
                    "and the problem is infeasible.")
    A = _remove_zero_rows(A)

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    d = []                  # Indices of dependent rows
    perm_r = None

    A_orig = A
    A = np.zeros((m, m + n), order='F')
    np.fill_diagonal(A, 1)
    A[:, m:] = A_orig
    e = np.zeros(m)

    js_candidates = np.arange(m, m+n, dtype=int)  # candidate columns for basis
    # manual masking was faster than masked array
    js_mask = np.ones(js_candidates.shape, dtype=bool)

    # Implements basic algorithm from [2]
    # Uses some of the suggested improvements (removing zero rows and
    # Bartels-Golub update idea).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis are only really useful if
    # the matrix is sparse.

    lu = np.eye(m, order='F'), np.arange(m)  # initial LU is trivial
    perm_r = lu[1]
    for i in v:

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        try:  # fails for i==0 and any time it gets ill-conditioned
            j = b[i-1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
        except Exception:
            lu = scipy.linalg.lu_factor(A[:, b])
            LU, p = lu
            perm_r = list(range(m))
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        js = js_candidates[js_mask]
        batch = 50

        # This is a tiny bit faster than looping over columns indivually,
        # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
        for j_index in range(0, len(js), batch):
            j_indices = js[j_index: min(j_index+batch, len(js))]

            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]  # very independent column
                b[i] = j
                js_mask[j-m] = False
                break
        else:
            d.append(i)
            if true_rank is not None and len(d) == m - true_rank:
                break   # found all redundancies

    keep = set(range(m))
    keep = list(keep - set(d))
    return A_orig[keep, :], keep