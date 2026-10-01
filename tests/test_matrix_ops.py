"""
Unit tests for src/matrix_ops.py.

Covers all public functions; validates algebraic identities from Magnus & Neudecker (1988)
that are relied upon by the CAW model implementation.
"""
import numpy as np
import pytest
from matrix_ops import (
    vech, vec, vech_to_matrix,
    duplication_matrix, elimination_matrix, commutation_matrix,
    is_positive_definite, nearest_positive_definite, safe_logdet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n)
    return (M + M.T) / 2


def _random_sym(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


# ---------------------------------------------------------------------------
# vech
# ---------------------------------------------------------------------------

class TestVech:
    def test_length(self):
        for n in [2, 3, 5]:
            M = np.eye(n)
            assert len(vech(M)) == n * (n + 1) // 2

    def test_identity_values(self):
        """vech of identity matrix should have 1 at diagonal positions, 0 elsewhere."""
        v = vech(np.eye(5))
        assert v[0]  == 1.0  # (0,0)
        assert v[5]  == 1.0  # (1,1)
        assert v[9]  == 1.0  # (2,2)
        assert v[12] == 1.0  # (3,3)
        assert v[14] == 1.0  # (4,4)
        # off-diagonal entries should all be zero
        assert np.all(v[[1, 2, 3, 4, 6, 7, 8, 10, 11, 13]] == 0.0)

    def test_known_ordering_n2(self):
        """For n=2: vech = [a11, a21, a22] (column-major lower triangle)."""
        M = np.array([[1., 3.], [3., 2.]])
        v = vech(M)
        np.testing.assert_array_equal(v, [1., 3., 2.])

    def test_n5_paper_ordering(self):
        """Paper vech order: r11,r21,r31,r41,r51,r22,r32,r42,r52,r33,r43,r53,r44,r54,r55."""
        M = np.arange(1, 26, dtype=float).reshape(5, 5)
        M = (M + M.T) / 2  # symmetrise
        v = vech(M)
        assert len(v) == 15
        # first column lower triangle: M[0,0], M[1,0], M[2,0], M[3,0], M[4,0]
        assert v[0] == M[0, 0]
        assert v[1] == M[1, 0]
        assert v[4] == M[4, 0]
        # second column: M[1,1], M[2,1], M[3,1], M[4,1]
        assert v[5] == M[1, 1]
        assert v[6] == M[2, 1]

    def test_single_element(self):
        M = np.array([[7.0]])
        np.testing.assert_array_equal(vech(M), [7.0])


# ---------------------------------------------------------------------------
# vec
# ---------------------------------------------------------------------------

class TestVec:
    def test_length(self):
        M = np.eye(5)
        assert len(vec(M)) == 25

    def test_column_major_ordering(self):
        """vec([[a,b],[c,d]]) = [a, c, b, d] in Fortran (column-major) order."""
        M = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_array_equal(vec(M), [1., 3., 2., 4.])

    def test_identity_5x5(self):
        v = vec(np.eye(5))
        expected = np.zeros(25)
        expected[[0, 6, 12, 18, 24]] = 1.0
        np.testing.assert_array_equal(v, expected)


# ---------------------------------------------------------------------------
# vech_to_matrix (inverse of vech)
# ---------------------------------------------------------------------------

class TestVechToMatrix:
    def test_roundtrip_sym_matrix(self):
        """vech_to_matrix(vech(M), n) should recover M for symmetric M."""
        for n in [2, 3, 5]:
            M = _random_sym(n)
            v = vech(M)
            M_rec = vech_to_matrix(v, n)
            np.testing.assert_allclose(M_rec, M, atol=1e-12)

    def test_output_is_symmetric(self):
        v = np.arange(1, 16, dtype=float)  # 15 elements for n=5
        M = vech_to_matrix(v, 5)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_output_shape(self):
        for n in [2, 3, 5]:
            k = n * (n + 1) // 2
            v = np.ones(k)
            M = vech_to_matrix(v, n)
            assert M.shape == (n, n)

    def test_single_element(self):
        np.testing.assert_allclose(vech_to_matrix(np.array([3.0]), 1), [[3.0]])


# ---------------------------------------------------------------------------
# duplication_matrix
# ---------------------------------------------------------------------------

class TestDuplicationMatrix:
    def test_shape(self):
        for n in [2, 3, 5]:
            D = duplication_matrix(n)
            assert D.shape == (n * n, n * (n + 1) // 2)

    def test_identity_dn_vech_equals_vec(self):
        """Core property: D_n @ vech(A) = vec(A) for symmetric A."""
        for n in [2, 3, 5]:
            D = duplication_matrix(n)
            M = _random_sym(n)
            np.testing.assert_allclose(D @ vech(M), vec(M), atol=1e-12)

    def test_binary_matrix(self):
        """D_n should contain only 0s and 1s."""
        D = duplication_matrix(5)
        assert set(np.unique(D)).issubset({0.0, 1.0})

    def test_n1_trivial(self):
        """n=1: D is a (1,1) identity."""
        D = duplication_matrix(1)
        assert D.shape == (1, 1)
        assert D[0, 0] == 1.0

    def test_each_column_has_at_most_two_ones(self):
        """Each vech element maps to at most 2 vec positions (diagonal only 1)."""
        D = duplication_matrix(5)
        col_sums = D.sum(axis=0)
        # diagonal entries map to exactly 1 row, off-diagonals to 2
        assert (col_sums <= 2).all()
        assert (col_sums >= 1).all()


# ---------------------------------------------------------------------------
# elimination_matrix
# ---------------------------------------------------------------------------

class TestEliminationMatrix:
    def test_shape(self):
        for n in [2, 3, 5]:
            L = elimination_matrix(n)
            assert L.shape == (n * (n + 1) // 2, n * n)

    def test_identity_ln_vec_equals_vech(self):
        """Core property: L_n @ vec(A) = vech(A) for symmetric A."""
        for n in [2, 3, 5]:
            L = elimination_matrix(n)
            M = _random_sym(n)
            np.testing.assert_allclose(L @ vec(M), vech(M), atol=1e-12)

    def test_ln_dn_product_is_identity(self):
        """L_n D_n = I_{k} for k = n*(n+1)/2."""
        for n in [2, 3, 5]:
            L = elimination_matrix(n)
            D = duplication_matrix(n)
            k = n * (n + 1) // 2
            np.testing.assert_allclose(L @ D, np.eye(k), atol=1e-12)


# ---------------------------------------------------------------------------
# commutation_matrix
# ---------------------------------------------------------------------------

class TestCommutationMatrix:
    def test_shape(self):
        for n in [2, 3, 5]:
            K = commutation_matrix(n)
            assert K.shape == (n * n, n * n)

    def test_identity_knn_vec_equals_vec_transpose(self):
        """Core property: K_nn @ vec(A) = vec(A') for any A."""
        for n in [2, 3, 5]:
            K = commutation_matrix(n)
            rng = np.random.default_rng(n)
            A = rng.standard_normal((n, n))
            np.testing.assert_allclose(K @ vec(A), vec(A.T), atol=1e-12)

    def test_knn_is_involution(self):
        """K_nn @ K_nn = I_{n^2}."""
        for n in [2, 3, 5]:
            K = commutation_matrix(n)
            np.testing.assert_allclose(K @ K, np.eye(n * n), atol=1e-12)

    def test_knn_is_orthogonal(self):
        """K_nn.T = K_nn^{-1} (permutation matrix ⟹ orthogonal)."""
        for n in [2, 3, 5]:
            K = commutation_matrix(n)
            np.testing.assert_allclose(K @ K.T, np.eye(n * n), atol=1e-12)

    def test_n1_trivial(self):
        K = commutation_matrix(1)
        np.testing.assert_allclose(K, [[1.0]])


# ---------------------------------------------------------------------------
# is_positive_definite
# ---------------------------------------------------------------------------

class TestIsPositiveDefinite:
    def test_spd_matrix_returns_true(self):
        assert is_positive_definite(_random_spd(5)) is True

    def test_identity_is_pd(self):
        assert is_positive_definite(np.eye(5)) is True

    def test_singular_matrix_returns_false(self):
        M = np.zeros((3, 3))
        assert is_positive_definite(M) is False

    def test_negative_definite_returns_false(self):
        assert is_positive_definite(-np.eye(3)) is False

    def test_single_positive_scalar(self):
        assert is_positive_definite(np.array([[2.0]])) is True

    def test_single_negative_scalar(self):
        assert is_positive_definite(np.array([[-1.0]])) is False


# ---------------------------------------------------------------------------
# nearest_positive_definite
# ---------------------------------------------------------------------------

class TestNearestPositiveDefinite:
    def test_already_pd_is_unchanged_approximately(self):
        M = _random_spd(4)
        M_pd = nearest_positive_definite(M)
        np.testing.assert_allclose(M_pd, M, atol=1e-8)

    def test_result_is_positive_definite(self):
        # Near-singular matrix
        M = np.ones((4, 4)) * 0.999 + np.eye(4) * 0.001
        M_pd = nearest_positive_definite(M)
        assert is_positive_definite(M_pd)

    def test_result_is_symmetric(self):
        M = np.array([[4., -1.], [-1., 0.]])  # not PD
        M_pd = nearest_positive_definite(M)
        np.testing.assert_allclose(M_pd, M_pd.T, atol=1e-12)

    def test_zero_matrix_becomes_pd(self):
        M_pd = nearest_positive_definite(np.zeros((3, 3)))
        assert is_positive_definite(M_pd)


# ---------------------------------------------------------------------------
# safe_logdet
# ---------------------------------------------------------------------------

class TestSafeLogdet:
    def test_identity_log_det_zero(self):
        assert abs(safe_logdet(np.eye(5))) < 1e-12

    def test_known_value(self):
        """det(diag(1, 2, 3, 4, 5)) = 120 → log = log(120)."""
        M = np.diag([1., 2., 3., 4., 5.])
        np.testing.assert_allclose(safe_logdet(M), np.log(120.), atol=1e-10)

    def test_spd_matrix_positive(self):
        M = _random_spd(5)
        result = safe_logdet(M)
        assert np.isfinite(result)
        assert result > -1e10

    def test_singular_raises(self):
        with pytest.raises(ValueError):
            safe_logdet(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Mathematical consistency between operators
# ---------------------------------------------------------------------------

class TestOperatorConsistency:
    """Cross-checks that the three operators are mutually consistent."""

    def test_dn_ln_is_projector_on_symmetric_vec(self):
        """D_n L_n acting on vec(A) for symmetric A should be identity on that space."""
        for n in [3, 5]:
            D = duplication_matrix(n)
            L = elimination_matrix(n)
            M = _random_sym(n)
            # D_n L_n vec(A) = vec(A) only for symmetric matrices
            np.testing.assert_allclose(D @ L @ vec(M), vec(M), atol=1e-12)

    def test_triple_relation_knn_dn_dn(self):
        """K_nn D_n = D_n (permutation symmetry for symmetric A)."""
        for n in [3, 5]:
            K = commutation_matrix(n)
            D = duplication_matrix(n)
            # K D = D because K permutes vec in a way that maps sym matrices to themselves
            np.testing.assert_allclose(K @ D, D, atol=1e-12)

    def test_vech_roundtrip_via_dn_ln(self):
        """L_n D_n vech(A) = vech(A)."""
        for n in [3, 5]:
            D = duplication_matrix(n)
            L = elimination_matrix(n)
            M = _random_sym(n)
            v = vech(M)
            np.testing.assert_allclose(L @ D @ v, v, atol=1e-12)
