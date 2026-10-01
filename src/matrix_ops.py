"""
Matrix operators for CAW realized covariance models.

Implements vech, vec, duplication, elimination, and commutation matrices
following the notation in Chiriac & Voev (2010) and Magnus & Neudecker (1988).
All operators use column-major (Fortran) ordering consistent with vec notation.
"""
import numpy as np
from typing import Tuple


def vech(matrix: np.ndarray) -> np.ndarray:
    """Extract lower-triangular elements of symmetric matrix into vector.

    Traverses column by column: for column j, extracts rows j through n-1.
    For n=5: r11, r21, r31, r41, r51, r22, r32, r42, r52, r33, r43, r53, r44, r54, r55

    Parameters
    ----------
    matrix : ndarray of shape (n, n)

    Returns
    -------
    ndarray of length n*(n+1)/2
    """
    n = matrix.shape[0]
    result = []
    for j in range(n):
        for i in range(j, n):
            result.append(matrix[i, j])
    return np.array(result)


def vec(matrix: np.ndarray) -> np.ndarray:
    """Column-major (Fortran-order) vectorization of matrix.

    Parameters
    ----------
    matrix : ndarray of shape (m, n)

    Returns
    -------
    ndarray of length m*n; element at (i,j) maps to position j*m+i
    """
    return matrix.flatten(order='F')


def vech_to_matrix(v: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct symmetric matrix from vech vector.

    Fills lower triangle column by column, then mirrors to upper triangle.

    Parameters
    ----------
    v : ndarray of length n*(n+1)/2
    n : int — matrix dimension

    Returns
    -------
    ndarray of shape (n, n), symmetric
    """
    matrix = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j, n):
            matrix[i, j] = v[idx]
            if i != j:
                matrix[j, i] = v[idx]
            idx += 1
    return matrix


def duplication_matrix(n: int) -> np.ndarray:
    """Construct duplication matrix D_n such that D_n @ vech(A) = vec(A) for symmetric A.

    D_n has shape (n^2, n*(n+1)/2).  For each lower-triangle element (i,j) with i>=j:
    - Sets D[j*n+i, p] = 1  (column-major position of A[i,j])
    - Sets D[i*n+j, p] = 1  (column-major position of A[j,i], if i != j)
    where p is the vech index of element (i,j).

    Parameters
    ----------
    n : int — matrix dimension

    Returns
    -------
    ndarray of shape (n^2, n*(n+1)/2)
    """
    k = n * (n + 1) // 2
    D = np.zeros((n * n, k), dtype=float)
    p = 0  # vech index
    for j in range(n):
        for i in range(j, n):
            D[j * n + i, p] = 1.0          # position of A[i,j] in vec(A)
            if i != j:
                D[i * n + j, p] = 1.0      # position of A[j,i] = A[i,j] in vec(A)
            p += 1
    return D


def elimination_matrix(n: int) -> np.ndarray:
    """Construct elimination matrix L_n such that L_n @ vec(A) = vech(A).

    L_n has shape (n*(n+1)/2, n^2).  For each lower-triangle element (i,j) with i>=j
    at vech index p, sets L[p, j*n+i] = 1 (column-major position of A[i,j] in vec(A)).

    Parameters
    ----------
    n : int — matrix dimension

    Returns
    -------
    ndarray of shape (n*(n+1)/2, n^2)
    """
    k = n * (n + 1) // 2
    L = np.zeros((k, n * n), dtype=float)
    p = 0  # vech index
    for j in range(n):
        for i in range(j, n):
            L[p, j * n + i] = 1.0          # column-major position of A[i,j]
            p += 1
    return L


def commutation_matrix(n: int) -> np.ndarray:
    """Construct commutation matrix K_nn such that K_nn @ vec(A) = vec(A').

    K_nn has shape (n^2, n^2).  Element K[j*n+i, i*n+j] = 1 for all i, j in [0,n),
    which swaps the row/column indices in the column-major layout:
        (K @ vec(A))[j*n+i]  =  vec(A)[i*n+j]  =  A[j,i]  =  A'[i,j]

    Parameters
    ----------
    n : int — matrix dimension

    Returns
    -------
    ndarray of shape (n^2, n^2)
    """
    K = np.zeros((n * n, n * n), dtype=float)
    for i in range(n):
        for j in range(n):
            K[j * n + i, i * n + j] = 1.0
    return K


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if matrix is positive definite via Cholesky decomposition.

    Parameters
    ----------
    matrix : ndarray of shape (n, n)

    Returns
    -------
    bool — True if Cholesky succeeds (all eigenvalues strictly positive)
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Project matrix to nearest positive definite matrix (Higham 1988).

    Algorithm:
    1. Symmetrise:        B = (A + A') / 2
    2. Polar factor:      H = V' diag(s) V  from SVD of B
    3. Candidate:         A_hat = (B + H) / 2, re-symmetrised
    4. Perturb diagonal if still not PD (numerical safety net).

    Parameters
    ----------
    matrix : ndarray of shape (n, n)

    Returns
    -------
    ndarray of shape (n, n), symmetric positive definite
    """
    B = (matrix + matrix.T) / 2
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt           # symmetric positive semidefinite polar factor
    A_hat = (B + H) / 2
    A_hat = (A_hat + A_hat.T) / 2        # re-symmetrise for floating-point safety

    # Iterative diagonal perturbation in case of residual numerical issues
    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(A_hat):
        min_eig = np.min(np.real(np.linalg.eigvals(A_hat)))
        A_hat += I * (-min_eig * k ** 2 + spacing)
        k += 1

    return A_hat


def safe_logdet(matrix: np.ndarray) -> float:
    """Compute log-determinant of matrix using numpy's sign-stable slogdet.

    Parameters
    ----------
    matrix : ndarray of shape (n, n), assumed positive definite

    Returns
    -------
    float — log|det(matrix)|; raises ValueError if det is non-positive
    """
    sign, logabsdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError(f"Matrix has non-positive determinant (sign={sign}).")
    return float(logabsdet)
