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
        m, n = K.XY.shape
        return (
            (K.XX.sum() - K.XX_trace()) / (m * (m - 1))
            + (K.YY.sum() - K.YY_trace()) / (n * (n - 1))
            - 2 * K.XY.mean()
        )
    elif estimator == Estimator.U_STAT:
        m, n = K.XY.shape
        assert m == n
        return (
            (K.XX.sum() - K.XX_trace())
            + (K.YY.sum() - K.YY_trace())
            - 2 * (K.XY.sum() - K.XY.trace())
        ) / (n * (n - 1))
    else:
        raise ValueError(f"unknown estimator type '{estimator}'")


def mmd2_permutations(kernel_pair, estimator=Estimator.U_STAT, permutations=500):
    K = kernel_pair.joint
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
    # XXX not tested yet!
    if estimator != Estimator.U_STAT:
        warnings.warn(
            "Computing asymptotic variance for U-statistic estimator, "
            f"but using {estimator}."
        )
    assert estimator == Estimator.U_STAT
    assert kernel_pair.n_X == kernel_pair.n_Y
    m = kernel_pair.n_X
    K = kernel_pair

    # Various kernel sums that we'll use to compute (5) of
    #   https://arxiv.org/abs/1611.04488v4
    # Kt means K with the diagonal set to 0, but no need to explicitly compute

    def sq_fro(a):
        a = a.view(-1)
        return a @ a

    if K.const_diagonal is False:
        diag_X = torch.diagonal(K.XX)
        diag_Y = torch.diagonal(K.YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = sq_fro(diag_X)
        sum_diag2_Y = sq_fro(diag_Y)
    else:
        v = torch.as_tensor(K.const_diagonal, dtype=K.XX.dtype, device=K.XX.device)
        diag_X = diag_Y = v[None]
        sum_diag_X = sum_diag_Y = m * v
        sum_diag2_X = sum_diag2_Y = m * v ** 2

    Kt_XX_sums = K.XX.sum(1) - diag_X
    Kt_YY_sums = K.YY.sum(1) - diag_Y
    K_XY_sums_0 = K.XY.sum(0)
    K_XY_sums_1 = K.XY.sum(1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    Kt_XX_2_sum = sq_fro(K.XX) - sum_diag2_X
    Kt_YY_2_sum = sq_fro(K.YY) - sum_diag2_Y
    K_XY_2_sum = sq_fro(K.XY)

    mm = m * m
    mmm = mm * m
    m1 = m - 1
    m_m1 = m * m1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1

    if estimator == Estimator.BIASED:
        mmd2 = (Kt_XX_sum + sum_diag_X + Kt_YY_sum + sum_diag_Y - 2 * K_XY_sum) / mm
    elif estimator == Estimator.UNBIASED:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / m_m1 - 2 * K_XY_sum / mm
    elif estimator == Estimator.U_STAT:
        mmd2 = (Kt_XX_sum + Kt_YY_sum - 2 * (K_XY_sum - K.XY.trace())) / m_m1
    else:
        raise ValueError(f"unknown estimator type '{estimator}'")

    var_est = (
        (
            (2 / (mm * m1_m1))
            * (
                2 * sq_fro(Kt_XX_sums)
                - Kt_XX_2_sum
                + 2 * sq_fro(Kt_YY_sums)
                - Kt_YY_2_sum
            )
        )
        - ((4 * m - 6) / (mmm * m1_m1_m1)) * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + (4 * (m - 2) / (mmm * m1_m1)) * (sq_fro(K_XY_sums_1) + sq_fro(K_XY_sums_0))
        - (4 * (m - 3) / (mmm * m1_m1)) * K_XY_2_sum
        - ((8 * m - 12) / (mm * mmm * m1)) * K_XY_sum ** 2
        + (
            (8 / (mmm * m1))
            * (
                1 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                - Kt_XX_sums @ K_XY_sums_1
                - Kt_YY_sums @ K_XY_sums_0
            )
        )
    )

    return mmd2, var_est
