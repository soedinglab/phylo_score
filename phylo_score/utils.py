import numpy as np
import scipy

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

def calc_scores(ind_ll, pair_ll, ind_reg, pair_reg, type : str):
    if type == "lldiff":
        return pair_ll - ind_ll
    elif type == "optdiff":
        return pair_ll - ind_ll - pair_reg + ind_reg
    else:
        raise ValueError(f"Unknown score type: {type}")
    
def calc_correction(scores, type : str):
    if type == "APC":
        return score_lin(average_product_correction(score_matrix(scores)))
    elif type == "ASC":
        return score_lin(average_sum_correction(score_matrix(scores)))
    elif type == "None":
        return scores
    else:
        raise ValueError(f"Unknown correction type: {type}")

def p_values(scores):

    assert(scores == scores).all()
    
    sort_scores = np.sort(scores)

    L = reverse_triangle_number(len(scores)) + 1

    # Remove L//2 highest scores, because they might be positive
    sort_scores = sort_scores[:-(L // 2)]


    fit = scipy.stats.gamma.fit(sort_scores, method='mle')
    
    print("Gamma distribution fit parameters (shape, loc, scale):")
    print(fit)

    p_vals = scipy.stats.gamma.sf(scores, *fit)

    return p_vals


def p_values_with_null(scores, null_scores):

    assert(scores == scores).all()
    assert(null_scores == null_scores).all()

    fit = scipy.stats.gamma.fit(null_scores, method='mle')
    
    print("Gamma distribution fit parameters (shape, loc, scale) for null alignment:")
    print(fit)

    p_vals = scipy.stats.gamma.sf(scores, *fit)

    return p_vals