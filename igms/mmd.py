import enum
import warnings

import torch
import numpy as np


class Estimator(enum.Enum):
    BIASED = enum.auto()
    UNBIASED = enum.auto()
    U_STAT = U_STATISTIC = enum.auto()


def mmd2(kernel_pair, estimator=Estimator.UNBIASED):
    K = kernel_pair

    if estimator == Estimator.BIASED:
        return K.XX.mean() + K.YY.mean() - 2 * K.XY.mean()
    elif estimator == Estimator.UNBIASED:
        return K.XX.offdiag_mean() + K.YY.offdiag_mean() - 2 * K.XY.mean()
    elif estimator == Estimator.U_STAT:
        return K.XX.offdiag_mean() + K.YY.offdiag_mean() - 2 * K.XY.offdiag_mean()
    else:
        raise ValueError(f"unknown estimator type '{estimator}'")


def mmd2_permutations(kernel_pair, estimator=Estimator.U_STAT, permutations=500):
    K = kernel_pair.joint()
    n_X = kernel_pair.n_X
    n_Y = kernel_pair.n_Y
    n = n_X + n_Y

    if estimator == Estimator.U_STAT:
        assert n_X == n_Y
        w_X = 1
        w_Y = -1
    elif estimator == Estimator.BIASED:
        w_X = 1 / n_X
        w_Y = -1 / n_Y
    elif estimator == Estimator.UNBIASED:
        msg = "haven't done permutations for UNBIASED yet; use U_STAT"
        raise NotImplementedError(msg)
    else:
        raise ValueError(f"unknown estimator type '{estimator}'")

    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, np.random.choice(n, n_X, replace=False)] = w_X

    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if estimator == Estimator.BIASED:
        ests = biased_ests
    elif estimator == Estimator.U_STAT:
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))

    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest


def mmd2_and_variance(kernel_pair, estimator=Estimator.U_STAT):
    if estimator != Estimator.U_STAT:
        warnings.warn(
            "Computing asymptotic variance for U-statistic estimator, "
            f"but using {estimator}."
        )
    assert estimator == Estimator.U_STAT
    assert kernel_pair.n_X == kernel_pair.n_Y
    m = kernel_pair.n_X
    K = kernel_pair

    # we're caching anyway
    mmd_est = mmd2(kernel_pair, estimator=estimator)

    mm = m * m
    mmm = mm * m
    m1 = m - 1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1

    # (5) of https://arxiv.org/abs/1611.04488v4
    var_est = (
        (
            (2 / (mm * m1_m1))
            * (
                2 * K.XX.offdiag_row_sums_sq_sum()
                - K.XX.offdiag_sq_sum()
                + 2 * K.YY.offdiag_row_sums_sq_sum()
                - K.YY.offdiag_sq_sum()
            )
        )
        - ((4 * m - 6) / (mmm * m1_m1_m1))
        * (K.XX.offdiag_sum() ** 2 + K.YY.offdiag_sum() ** 2)
        + (4 * (m - 2) / (mmm * m1_m1))
        * (K.XY.col_sums_sq_sum() + K.XY.row_sums_sq_sum())
        - (4 * (m - 3) / (mmm * m1_m1)) * K.XY.sq_sum()
        - ((8 * m - 12) / (mm * mmm * m1)) * K.XY.sum() ** 2
        + (
            (8 / (mmm * m1))
            * (
                1 / m * (K.XX.offdiag_sum() + K.YY.offdiag_sum()) * K.XY.sum()
                - K.XX.offdiag_row_sums() @ K.XY.col_sums()
                - K.YY.offdiag_row_sums() @ K.XY.row_sums()
            )
        )
    )

    return mmd_est, var_est
