"""
Some probably over-engineered infrastructure for lazily computing kernel
matrices, allowing for various sums / means / etc used by MMD-related estimators.
"""
from enum import Enum
from functools import partial, wraps
import inspect
from inspect import Parameter

import torch

from .utils import as_tensors, floats, float_or_none, floats_or_none


def _cache(f):
    cache_name = f.__name__

    @wraps(f)
    def wrapper(self):
        try:
            return self._cache[cache_name]
        except KeyError:
            val = f(self)
            self._cache[cache_name] = val
            return val

    return wrapper


################################################################################
# Matrix wrappers that cache sums / etc.


class Matrix:
    def __init__(self, M, const_diagonal=False):
        self.mat = M = torch.as_tensor(M)
        self.m, self.n = self.shape = M.shape
        self._cache = {}

    @_cache
    def row_sums(self):
        return self.mat.sum(0)

    @_cache
    def col_sums(self):
        return self.mat.sum(1)

    @_cache
    def row_sums_sq_sum(self):
        sums = self.row_sums()
        return sums @ sums

    @_cache
    def col_sums_sq_sum(self):
        sums = self.col_sums()
        return sums @ sums

    @_cache
    def sum(self):
        if "row_sums" in self._cache:
            return self.row_sums().sum()
        elif "col_sums" in self._cache:
            return self.col_sums().sum()
        else:
            return self.mat.sum()

    def mean(self):
        return self.sum() / (self.m * self.n)

    @_cache
    def sq_sum(self):
        flat = self.mat.view(-1)
        return flat @ flat

    def __repr__(self):
        return f"<{type(self).__name__}, {self.m} by {self.n}>"


class SquareMatrix(Matrix):
    def __init__(self, M):
        super().__init__(M)
        assert self.m == self.n

    @_cache
    def diagonal(self):
        return self.mat.diagonal()

    @_cache
    def trace(self):
        return self.mat.trace()

    @_cache
    def sq_trace(self):
        diag = self.diagonal()
        return diag @ diag

    @_cache
    def offdiag_row_sums(self):
        return self.row_sums() - self.diagonal()

    @_cache
    def offdiag_col_sums(self):
        return self.col_sums() - self.diagonal()

    @_cache
    def offdiag_row_sums_sq_sum(self):
        sums = self.offdiag_row_sums()
        return sums @ sums

    @_cache
    def offdiag_col_sums_sq_sum(self):
        sums = self.offdiag_col_sums()
        return sums @ sums

    @_cache
    def offdiag_sum(self):
        return self.offdiag_row_sums().sum()

    def offdiag_mean(self):
        return self.offdiag_sum() / (self.n * (self.n - 1))

    @_cache
    def offdiag_sq_sum(self):
        return self.sq_sum() - self.sq_trace()


class SymmetricMatrix(SquareMatrix):
    def col_sums(self):
        return self.row_sums()

    def sums(self):
        return self.row_sums()

    def offdiag_col_sums(self):
        return self.offdiag_row_sums()

    def offdiag_sums(self):
        return self.offdiag_row_sums()

    def col_sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def offdiag_col_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()

    def offdiag_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()


class ConstDiagMatrix(SquareMatrix):
    def __init__(self, M, diag_value):
        super().__init__(M)
        self.diag_value = diag_value

    @_cache
    def diagonal(self):
        return self.mat.new_full((1,), self.diag_value)

    def trace(self):
        return self.n * self.diag_value

    def sq_trace(self):
        return self.n * (self.diag_value ** 2)


class SymmetricConstDiagMatrix(ConstDiagMatrix, SymmetricMatrix):
    pass


def as_matrix(M, const_diagonal=False, symmetric=False):
    if symmetric:
        if const_diagonal is not False:
            return SymmetricConstDiagMatrix(M, diag_value=const_diagonal)
        else:
            return SymmetricMatrix(M)
    elif const_diagonal is not False:
        return ConstDiagMatrix(M, diag_value=const_diagonal)
    elif M.shape[0] == M.shape[1]:
        return SquareMatrix(M)
    else:
        return Matrix(M)


################################################################################
# A function to choose kernel classes from a string spec.

_registry = {}
_skip_names = frozenset(("X", "Y", "n_X"))
_skip_types = frozenset((Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD))


def register(original_cls=None, *, name=None):
    def decorator(cls):
        sig = inspect.signature(cls)
        arg_info = []
        for kw, param in sig.parameters.items():
            if param.name in _skip_names or param.kind in _skip_types:
                continue
            parser = param.annotation
            arg_info.append((kw, str if parser is Parameter.empty else parser))

        _registry[(name or cls.__name__).lower()] = (cls, arg_info)
        return cls

    return decorator(original_cls) if original_cls else decorator


def pick_kernel(spec):
    name, *args = spec.split(":")

    name = "".join(x for x in name.split("_")).lower()

    fn, arg_info = _registry[name]
    assert len(args) <= len(arg_info)
    kwargs = {name: parser(s) for s, (name, parser) in zip(args, arg_info)}
    return partial(fn, **kwargs)


################################################################################
# Kernel base class

_StorageMode = Enum("_StorageMode", "BOTH TO_SELF STACKED")


class LazyKernelPair:
    def __init__(self, X, Y=None, n_X=None):
        self.X, Y = as_tensors(X, Y)

        if n_X is not None:
            assert Y is None
            self.n_X = n_X
            self.n_Y = self.X.shape[0] - n_X
            self._storage = _StorageMode.STACKED
        else:
            self.n_X = self.X.shape[0]
            if Y is None:
                self.n_Y = self.n_X
                self._storage = _StorageMode.TO_SELF
            else:
                self.Y = Y
                self.n_Y = self.Y.shape[0]
                self._storage = _StorageMode.BOTH

        self._cache = {}
        if not hasattr(self, "const_diagonal"):
            self.const_diagonal = False

    def __repr__(self):
        return f"<{type(self).__name__}({self.n_X} to {self.n_Y})>"

    def _precompute(self, A):
        return ()

    @_cache
    def _precompute_X(self):
        return self._precompute(self.X)

    @_cache
    def _precompute_Y(self):
        return self._precompute(self.Y)

    @_cache
    def _compute_stacked(self):
        big_info = self._precompute(self.X)
        return self._compute(self.X, self.X, *big_info, *big_info)

    @property
    @_cache
    def XX(self):
        if self._storage == _StorageMode.STACKED:
            res = self._compute_stacked()[: self.n_X, : self.n_X]
        else:
            X_info = self._precompute_X()
            res = self._compute(self.X, self.X, *X_info, *X_info)
        return as_matrix(res, const_diagonal=self.const_diagonal, symmetric=True)

    @property
    @_cache
    def YY(self):
        if self._storage == _StorageMode.TO_SELF:
            return self.XX
        elif self._storage == _StorageMode.STACKED:
            res = self._compute_stacked()[self.n_X :, self.n_X :]
        else:
            Y_info = self._precompute_Y()
            res = self._compute(self.Y, self.Y, *Y_info, *Y_info)
        return as_matrix(res, const_diagonal=self.const_diagonal, symmetric=True)

    @property
    @_cache
    def XY(self):
        if self._storage == _StorageMode.TO_SELF:
            return self.XX
        elif self._storage == _StorageMode.STACKED:
            res = self._compute_stacked()[: self.n_X, self.n_X :]
        else:
            X_info = self._precompute_X()
            Y_info = self._precompute_Y()
            res = self._compute(self.X, self.Y, *X_info, *Y_info)
        return as_matrix(res)

    @_cache
    def joint(self):
        if self._storage == _StorageMode.STACKED:
            return as_matrix(
                self._compute_stacked(), const_diagonal=self.const_diagonal
            )
        else:
            XX = self.XX.mat
            XY = self.XY.mat
            YY = self.YY.mat
            return torch.cat([torch.cat([XX, XY], 1), torch.cat([XY.t(), YY], 1)], 0)

    def as_tensors(self, *args, **kwargs):
        kwargs.setdefault("device", self.X.device)
        kwargs.setdefault("dtype", self.X.dtype)
        return tuple(None if r is None else torch.as_tensor(r, **kwargs) for r in args)


################################################################################
# Finally, the actual kernel functions.


@register
class Linear(LazyKernelPair):
    def _compute(self, A, B):
        return A @ B.t()


@register
class Polynomial(LazyKernelPair):
    def __init__(
        self,
        X,
        Y=None,
        n_X=None,
        degree: float = 3,
        gamma: float_or_none = None,
        coef0: float = 1,
    ):
        super().__init__(X, Y, n_X=n_X)
        self.degree = degree
        self.gamma = 1 / X.shape[1] if gamma is None else gamma
        self.coef0 = coef0

    def _compute(self, A, B):
        XY = A @ B.t()
        return (self.gamma * XY + self.coef0) ** self.degree


@register
class LinearAndSquare(LazyKernelPair):
    "k(X, Y) = <X, Y> + w <X^2, Y^2>, with the squaring elementwise."

    def __init__(self, X, Y=None, n_X=None, w: float = 1):
        super().__init__(X, Y, n_X=n_X)
        self.w = w

    def _precompute(self, A):
        return (A * A,)

    def _compute(self, A, B, A_squared, B_squared):
        return A @ A.t() + self.w * (A_squared @ B_squared.t())


@register
class MixRBFDot(LazyKernelPair):
    def __init__(
        self,
        X,
        Y=None,
        n_X=None,
        sigmas_sq: floats = (1,),
        wts: floats_or_none = None,
        add_dot: float = 0,
    ):
        super().__init__(X, Y, n_X=n_X)
        self.sigmas_sq, self.wts = self.as_tensors(sigmas_sq, wts)
        self.add_dot = add_dot
        if not add_dot:
            self.const_diagonal = 1 if self.wts is None else self.wts.sum()

    def _precompute(self, A):
        return (torch.einsum("ij,ij->i", A, A),)

    def _compute(self, A, B, A_sqnorms, B_sqnorms):
        dot = A @ B.t()
        D2 = A_sqnorms[:, None] + B_sqnorms[None, :] - 2 * dot
        K_parts = torch.exp(D2[None, :, :] / (-2 * self.sigmas_sq[:, None, None]))
        if self.wts is None:
            K = K_parts.mean(0)
        else:
            K = torch.einsum("sij,s->ij", K_parts, self.wts)
        return (K + self.add_dot * dot) if self.add_dot else K


@register
class MixRBF(MixRBFDot):
    def __init__(
        self, X, Y=None, n_X=None, sigmas_sq: floats = (1,), wts: floats_or_none = None
    ):
        super().__init__(X, Y, n_X=n_X, sigmas_sq=sigmas_sq, wts=wts, add_dot=0)


@register
class RBF(MixRBFDot):
    def __init__(self, X, Y=None, n_X=None, sigma_sq: float = 1):
        super().__init__(X, Y, n_X=n_X, sigmas_sq=(sigma_sq,), wts=None, add_dot=0)
