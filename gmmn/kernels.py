from functools import partial

import torch

from .utils import as_tensors

# Some infrastructure, probably over-engineered, that lets us compute kernels
# in various ways.


class KernelPair:
    def __init__(self, K, XX=None, YY=None, n_X=None, const_diagonal=False):
        K, XX, YY = as_tensors(K, XX, YY)
        self.const_diagonal = const_diagonal

        if n_X is None or n_X == K.shape[0]:
            self.n_X, self.n_Y = K.shape
            self.XY = K
            self.YX = K.t()

            if XX is None:
                assert YY is None
            else:
                self.XX = XX
                self.YY = YY
        else:
            assert XX is None and YY is None
            self.n_X = n_X
            self.n = K.shape[0]
            assert K.shape[1] == self.n
            self.n_Y = self.n - self.n_X

            self._joint = K
            self.XX = K[:n_X, :n_X]
            self.XY = K[:n_X, n_X:]
            self.YX = K[n_X:, :n_X]
            self.YY = K[n_X:, n_X:]

    @property
    def joint(self):
        if not hasattr(self, "_joint"):
            self._joint = K = torch.cat(
                [torch.cat([self.XX, self.XY], 1), torch.cat([self.YX, self.YY], 1)], 0
            )

            n_X = self.n_X
            self.XX = K[:n_X, :n_X]
            self.XY = K[:n_X, n_X:]
            self.YX = K[n_X:, :n_X]
            self.YY = K[n_X:, n_X:]
        return self._joint


def _make_pair(K_XY, get_K_XX, get_K_YY, Y_none, n1, XY_only):
    if Y_none:
        if n1 is None:
            return KernelPair(K_XY, K_XY, K_XY)
        else:
            return KernelPair(K_XY, n_X=n1)
    elif XY_only:
        return KernelPair(K_XY)
    else:
        return KernelPair(K_XY, get_K_XX(), get_K_YY())


def linear(X, Y=None, n1=None, XY_only=False):
    X, Y = as_tensors(X, Y)
    XY = X @ (X if Y is None else Y).t()
    return _make_pair(XY, lambda: X @ X.t(), lambda: Y @ Y.t(), Y is None, n1, XY_only)


def mix_rbf_dot(X, Y=None, sigmas_sq=(1,), wts=None, add_dot=0, n1=None, XY_only=False):
    X, Y, wts, sigmas_sq = as_tensors(X, Y, wts, sigmas_sq)

    X_sqnorms = torch.einsum("ij,ij->i", X, X)

    if Y is None:
        Y_sqnorms = X_sqnorms
    else:
        Y_sqnorms = torch.einsum("ij,ij->i", Y, Y)

    def get_K(m1, m2, sqnorms1, sqnorms2):
        dot = m1 @ m2.t()
        D2 = sqnorms1[:, None] + sqnorms2[None, :] - 2 * dot
        K_parts = torch.exp(D2[None, :, :] / (-2 * sigmas_sq[:, None, None]))
        if wts is None:
            K = K_parts.mean()
        else:
            K = torch.einsum("sij,s->ij", K_parts, wts)
        return (K + add_dot * dot) if add_dot else K

    K_XY = get_K(X, Y, X_sqnorms, Y_sqnorms)
    get_K_XX = partial(get_K, X, X, X_sqnorms, Y_sqnorms)
    get_K_YY = partial(get_K, Y, Y, Y_sqnorms, Y_sqnorms)
    return _make_pair(K_XY, get_K_XX, get_K_YY, Y is None, n1, XY_only)
