import enum
import warnings

import torch
import numpy as np

from .utils import fill_diagonal


class Estimator(enum.Enum):
    BIASED = enum.auto()
    UNBIASED = enum.auto()
    U_STAT = U_STATISTIC = enum.auto()


def mmd2(K, estimator=Estimator.UNBIASED, inds=(0, 1)):
    """
    Estimate the squared MMD.

    - K: a LazyKernelResult object.
    - inds: which parts of K to compute on [default (0, 1)].

    By default, we estimate the MMD between K.parts[0] and K.parts[1].
    """
    i, j = inds

    XX = K.matrix(i, i)
    XY = K.matrix(i, j)
    YY = K.matrix(j, j)

    if estimator == Estimator.BIASED:
        return XX.mean() + YY.mean() - 2 * XY.mean()
    elif estimator == Estimator.UNBIASED:
        return XX.offdiag_mean() + YY.offdiag_mean() - 2 * XY.mean()
    elif estimator == Estimator.U_STAT:
        assert XY.m == XY.n
        return XX.offdiag_mean() + YY.offdiag_mean() - 2 * XY.offdiag_mean()
    else:
        raise ValueError(f"unknown estimator type '{estimator}'")


def pairwise_mmd2s(
    K, estimator=Estimator.UNBIASED, use_joint=True, inds=None, skip_diag=True
):
    """
    Computes the matrix of all pairwise mmd2s for the given LazyKernelResult.

    If use_joint is False, does the computations separately (more memory-efficent);
    if True, does them all together (possibly faster, depending).

    If skip_diag, just use 0s on the diagonal. The unbiased estimator will
    otherwise be negative.
    """
    if inds is None:
        inds = range(K.n_parts)

    if not use_joint:
        return torch.stack(
            [
                0 if skip_diag and i == j else mmd2(K, estimator=estimator, inds=(i, j))
                for i in inds
                for j in inds
            ],
            0,
        ).reshape(K.n_parts, K.n_parts)

    if estimator not in {Estimator.BIASED, Estimator.UNBIASED, Estimator.U_STAT}:
        raise ValueError(f"unknown estimator type '{estimator}'")

    if estimator == Estimator.U_STAT:
        assert len(set(K.ns)) == 1
        raise NotImplementedError(
            "haven't implemented u-statistic est with use_joint=True"
        )

    joint = K.joint(*inds)

    # probably a more efficient way to do this
    wts = joint.new_zeros(K.n_parts, joint.shape[0])
    start = 0
    for i, n in enumerate(K.ns):
        wts[i, start : start + n] = 1 / n
        start += n

    block_means = torch.einsum("ia,jb,ab->ij", wts, wts, joint)

    if estimator == Estimator.BIASED:
        diag = torch.diagonal(block_means)
        est = diag[:, None] + diag[None, :] - 2 * block_means

    elif estimator == Estimator.UNBIASED:
        # have to correct the mean-to-self terms...
        # subtract off the sum of the diagonal elements
        ns = torch.as_tensor(K.ns, dtype=joint.dtype, device=joint.device)
        if K.const_diagonal:
            diag_sums = K.const_diagonal / ns
        else:
            diag_sums = torch.einsum("ia,ia,aa->i", wts, wts, joint)
        # and multiply by n^2 / (n (n-1)) = n / (n-1)
        corrs = ns / (ns - 1.0)
        unbiased_diag = (torch.diagonal(block_means) - diag_sums) * corrs

        est = unbiased_diag[None, :] + unbiased_diag[:, None] - 2 * block_means

    return fill_diagonal(est, 0) if skip_diag else est


def mmd2_permutations(K, estimator=Estimator.U_STAT, permutations=500, inds=(0, 1)):
    i, j = inds
    n_X = K.n(i)
    n_Y = K.n(j)
    K = K.joint(i, j)
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


def mmd2_and_unbiased_variance(K, estimator=Estimator.U_STAT, inds=(0, 1)):
    """
    Estimate MMD variance with the unbiased estimator
    from https://arxiv.org/abs/1906.02104.
    """
    if estimator != Estimator.U_STAT:
        warnings.warn(
            "Estimating variance for the U-statistic estimator, "
            f"but using {estimator}."
        )
    i, j = inds
    m = K.n(i)
    assert K.n(j) == m
    XX = K.matrix(i, i)
    XY = K.matrix(i, j)
    YY = K.matrix(j, j)

    # we're caching anyway
    mmd_est = mmd2(K, estimator=estimator, inds=inds)

    mm = m * m
    mmm = mm * m
    m1 = m - 1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1
    m2 = m - 2
    mdown2 = m * m1
    mdown3 = mdown2 * m2
    mdown4 = mdown3 * (m - 3)
    twom3 = 2 * m - 3

    var_est = (
        (4 / mdown4) * (XX.offdiag_sums_sq_sum() + YY.offdiag_sums_sq_sum())
        + (4 * (mm - m - 1) / (mmm * m1_m1))
        * (XY.row_sums_sq_sum() + XY.col_sums_sq_sum())
        - (8 / (mm * (mm - 3 * m + 2)))
        * (XX.offdiag_sums() @ XY.col_sums() + YY.offdiag_sums() @ XY.row_sums())
        + 8 / (mm * mdown3) * ((XX.offdiag_sum() + YY.offdiag_sum()) * XY.sum())
        - (2 * twom3 / (mdown2 * mdown4)) * (XX.offdiag_sum() + YY.offdiag_sum())
        - (4 * twom3 / (mmm * m1_m1_m1)) * XY.sum() ** 2
        - (2 / (m * (mmm - 6 * mm + 11 * m - 6)))
        * (XX.offdiag_sq_sum() + YY.offdiag_sq_sum())
        + (4 * m2 / (mm * m1_m1_m1)) * XY.sq_sum()
    )

    return mmd_est, var_est


def mmd2_and_vstat_variance(K, estimator=Estimator.U_STAT, inds=(0, 1)):
    """
    Estimate the asymptotic MMD variance with a simple V-statistic.
    """
    i, j = inds
    m = K.n(i)
    assert K.n(j) == m
    XX = K.matrix(i, i)
    XY = K.matrix(i, j)
    YY = K.matrix(j, j)

    # we're caching anyway
    mmd_est = mmd2(K, estimator=estimator, inds=inds)

    # H = XX + YY - XY - XY.t()
    # estimator is
    #   4/m * [ mean(col_means(H) ** 2) - mean(H)**2 ]
    H_col_means = (XX.col_sums() + YY.col_sums() - XY.col_sums() - XY.row_sums()) / m
    var_est = (
        4
        / m
        * (
            (H_col_means @ H_col_means) / m
            - (XX.sum() + YY.sum() - 2 * XY.sum()) ** 2 / m ** 4
        )
    )
    return mmd_est, var_est


def diff_mmd2_and_variance(K, estimator=Estimator.U_STAT, inds=(0, 1, 2)):
    """
    Core components of a test for whether MMD(X, Y) < MMD(X, Z),
    assuming distributions of X, Y, Z are all distinct. Uses the U-statistic
    estimator and its asymptotic variance.

    Estimator from https://arxiv.org/abs/1906.02104.
    """

    if estimator != Estimator.U_STAT:
        warnings.warn(
            "Computing asymptotic variance for U-statistic estimator, "
            f"but using {estimator}."
        )

    i, j, k = inds
    m = K.n(i)
    assert m == K.n(j) == K.n(k)

    mmd2_XY = mmd2(K, estimator=estimator, inds=(i, j))
    mmd2_XZ = mmd2(K, estimator=estimator, inds=(i, k))
    diff_est = mmd2_XY - mmd2_XZ

    XY = K.matrix(i, j)
    XZ = K.matrix(i, k)
    YY = K.matrix(j, j)
    ZZ = K.matrix(k, k)

    m = XY.m
    mm = m * m
    mmm = mm * m
    mmmm = mmm * m

    m1 = m - 1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1

    m2 = m - 2

    mdown2 = m * m1
    mdown3 = mdown2 * m2
    mdown4 = mdown3 * (m - 3)

    twom3 = 2 * m - 3

    var_est = (
        (4 * (mm - m - 1) / (mmm * m1_m1_m1))
        * (
            XY.row_sums_sq_sum()
            + XY.col_sums_sq_sum()
            + XZ.row_sums_sq_sum()
            + XZ.col_sums_sq_sum()
        )
        + (4 / mdown4) * (YY.sums_sq_sum() + ZZ.sums_sq_sum())
        - (8 / mmm * m1) * (XY.col_sums() @ XZ.col_sums())
        - (8 / (mm * (mm - 3 * m + 2)))
        * (
            YY.offdiag_row_sums() @ XY.row_sums()
            + ZZ.offdiag_row_sums() @ XZ.row_sums()
        )
        - (4 * twom3 / (mmm * m1_m1_m1)) * (XY.sum() ** 2 + XZ.sum() ** 2)
        - (2 * twom3 / (mdown2 * mdown4))
        * (YY.offdiag_sum() ** 2 + ZZ.offdiag_sum() ** 2)
        + (8 / (mmmm * m1)) * (XY.sum() * XZ.sum())
        + (8 / (mm * mdown3)) * (YY.sum() * XY.sum() + ZZ.sum() * XZ.sum())
        - (4 * m2 / (mm * m1_m1_m1)) * (XY.sq_sum() + XZ.sq_sum())
        - (2 / (m * (mmm - 6 * mm + 11 * m - 6)))
        * (YY.offdiag_sq_sum() + ZZ.offdiag_sq_sum())
    )

    return diff_est, var_est
