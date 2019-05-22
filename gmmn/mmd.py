import enum

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
