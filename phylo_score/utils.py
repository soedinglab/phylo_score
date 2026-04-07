import numpy as np

def all_pairs_numpy(k : int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(k, k=1)

def reverse_triangle_number(n: int) -> int:
    """
    Given a number n, return the largest k such that k * (k + 1) / 2 <= n.
    This is the reverse of the triangle number sequence.
    """
    return (int(round((1 + 8 * n) ** 0.5)) - 1) // 2

def score_matrix(scores):
    """
        Calculates the score matrix from the flattened scores (upper triangular without diagonal)
    """
    L = reverse_triangle_number(len(scores)) + 1
    matrix = np.zeros([L,L])
    matrix[all_pairs_numpy(L)] = scores
    return matrix + matrix.transpose()

def score_lin(s_matrix):
    """
        s_matrix has shape (L,L) with all the scores (it is symmetric, the diagonal has to be zeroed)
    """
    assert((np.diag(s_matrix) == 0).all())
    assert((s_matrix == s_matrix.T).all())
    L = s_matrix.shape[0]
    return s_matrix[all_pairs_numpy(L)]

def average_product_correction(s_matrix):
    """
        s_matrix has shape (L,L) with all the scores (it is symmetric, the diagonal has to be zeroed)
    """
    assert((np.diag(s_matrix) == 0).all())
    assert((s_matrix == s_matrix).all())
    assert((s_matrix == s_matrix.T).all())
    L = s_matrix.shape[0]

    # In case some of the scores are negative, shift them all to be non-negative
    if s_matrix.min() < 0:
        s_matrix = s_matrix - s_matrix.min()

    cov_i = s_matrix.sum(axis = 0) / (L - 1)

    cov = cov_i.mean()

    apc = s_matrix - (cov_i[:, None] * cov_i[None, :]) / cov
    return apc - np.diag(np.diag(apc))

def average_sum_correction(s_matrix):
    """
        s_matrix has shape (L,L) with all the scores (it is symmetric, the diagonal has to be zeroed)
    """
    assert((np.diag(s_matrix) == 0).all())
    assert((s_matrix == s_matrix.T).all())
    L = s_matrix.shape[0]

    cov_i = s_matrix.sum(axis = 0) / (L - 1)

    cov = cov_i.mean()

    apc = s_matrix - (cov_i[:, None] + cov_i[None, :] - cov)
    return apc - np.diag(np.diag(apc))
