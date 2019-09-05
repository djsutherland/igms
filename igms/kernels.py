"""
Some probably over-engineered infrastructure for lazily computing kernel
matrices, allowing for various sums / means / etc used by MMD-related estimators.
"""
from functools import partial, wraps
import inspect
from inspect import Parameter

import numpy as np
import torch

from .utils import as_tensors, floats, float_or_none, floats_or_none


def _cache(f):
    # Only works when the function takes no or simple arguments!
    @wraps(f)
    def wrapper(self, *args):
        key = (f.__name__,) + tuple(args)
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = val = f(self, *args)
        return val

    return wrapper


################################################################################
# Matrix wrappers that cache sums / etc.


class Matrix:
    def __init__(self, M, const_diagonal=False):
        self.mat = M = torch.as_tensor(M)
        self.m, self.n = self.shape = M.shape
        self._cache = {}

    @property
    def dtype(self):
        return self.mat.dtype

    @property
    def device(self):
        return self.mat.device

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
        flat = self.mat.contiguous().view(-1)
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

# TODO: could make this more like a typical Module by making the class
#       basically just set up the kernel function arguments,
#       then return this class as results when you call it?
_name_map = {"X": 0, "Y": 1, "Z": 2}


class LazyKernel(torch.nn.Module):
    """
    Base class that allows computing kernel matrices among a bunch of datasets,
    only computing the matrices when we use them.

    Constructor arguments:
        - A bunch of matrices we'll compute the kernel among.
          2+d tensors, with all dimensions after the first agreeing,
          or else the special value None, meaning to use the first entry X.
          (This is more efficient than passing the same tensor again.)

    Access the results with:
      - K[0, 1] to get the Tensor between parts 0 and 1.
      - K.XX, K.XY, K.ZY, etc: shortcuts, with X meaning 0, Y 1, Z 2.
      - K.matrix(0, 1) or K.XY_m: returns a Matrix subclass.
    """

    def __init__(self, X, *rest):
        super().__init__()
        self._cache = {}
        if not hasattr(self, "const_diagonal"):
            self.const_diagonal = False

        # want to use pytorch buffer for parts
        # but can't assign a list to those, so munge some names
        X, *rest = as_tensors(X, *rest)
        if len(X.shape) < 2:
            raise ValueError(
                "LazyKernel expects inputs to be at least 2d. "
                "If your data is 1d, make it [n, 1] with X[:, np.newaxis]."
            )

        self.register_buffer("_part_0", X)
        self.n_parts = 1
        for p in rest:
            self.append_part(p)

    def __repr__(self):
        return f"<{type(self).__name__}({', '.join(str(n) for n in self.ns)})>"

    ############################################################################
    # The main interface to compute kernels

    def _compute(self, A, B):
        """
        Compute the kernel matrix between A and B.

        Might get called with A = X, B = X, or A = X, B = Y, etc.

        Should return a tensor of shape [A.shape[0], B.shape[0]].

        If you implement _precompute, this gets added to the signature here:
            self._compute(A, *self._precompute(A), B, *self._precompute(B)).
        The default _precompute returns an empty tuple, so it's _compute(A, B),
        but if you make a _precompute that returns [A_squared, A_cubed] then it's
            self._compute(A, A_squared, A_cubed, B, B_squared, B_cubed).
        """
        raise NotImplementedError("subclasses must implement _compute")

    def _precompute(self, A):
        """
        Compute something extra for each part A.

        Can be used to share computation between kernel(X, X) and kernel(X, Y).

        We end up calling basically (but with caching)
            self._compute(A, *self._precompute(A), B, *self._precompute(B))
        This default _precompute returns an empty tuple, so it's
            self._compute(A, B)
        But if you return [A_squared], it'd be
            self._compute(A, A_squared, B, B_squared)
        and so on.
        """
        return ()

    ############################################################################
    # Stuff that does the work of computing kernels

    @_cache
    def _precompute_i(self, i):
        p = self._part(i)
        if p is None:
            return self._precompute_i(0)
        return self._precompute(p)

    @_cache
    def __getitem__(self, k):
        try:
            i, j = k
        except ValueError:
            raise KeyError("You should index kernels with pairs")

        A = self._part(i)
        if A is None:
            return self[0, j]

        B = self._part(j)
        if B is None:
            return self[i, 0]

        if i > j:
            return self[j, i].t()

        A_info = self._precompute_i(i)
        B_info = self._precompute_i(j)
        return self._compute(A, *A_info, B, *B_info)

    @_cache
    def matrix(self, i, j):
        if self._part(i) is None:
            return self.matrix(0, j)

        if self._part(j) is None:
            return self.matrix(i, 0)

        k = self[i, j]
        if i == j:
            return as_matrix(k, const_diagonal=self.const_diagonal, symmetric=True)
        else:
            return as_matrix(k)

    @_cache
    def joint(self, *inds):
        if not inds:
            return self.joint(*range(self.n_parts))
        return torch.cat([torch.cat([self[i, j] for j in inds], 1) for i in inds], 0)

    @_cache
    def joint_m(self, *inds):
        if not inds:
            return self.joint_m(*range(self.n_parts))
        return as_matrix(
            self.joint(*inds), const_diagonal=self.const_diagonal, symmetric=True
        )

    ############################################################################
    # Helpers to access things more nicely

    def __getattr__(self, name):
        # self.X, self.Y, self.Z
        if name in _name_map:
            i = _name_map[name]
            if i < self.n_parts:
                return self.part(i)
            else:
                raise AttributeError(f"have {self.n_parts} parts, asked for {i}")

        # self.XX, self.XY, self.YZ, etc; also self.XX_m
        ret_matrix = False
        if len(name) == 4 and name.endswith("_m"):
            ret_matrix = True
            name = name[:2]

        if len(name) == 2:
            i = _name_map.get(name[0], np.inf)
            j = _name_map.get(name[1], np.inf)
            if i < self.n_parts and j < self.n_parts:
                return self.matrix(i, j) if ret_matrix else self[i, j]
            else:
                raise AttributeError(f"have {self.n_parts} parts, asked for {i}, {j}")

        return super().__getattr__(name)

    def _part(self, i):
        return self._buffers[f"_part_{i}"]

    def part(self, i):
        p = self._part(i)
        return self.X if p is None else p

    @property
    def parts(self):
        return [self.part(i) for i in range(self.n_parts)]

    def n(self, i):
        return self.part(i).shape[0]

    @property
    def ns(self):
        return [self.n(i) for i in range(self.n_parts)]

    ############################################################################
    # Stuff related to adding/removing data parts

    def _invalidate_cache(self, i):
        for k in list(self._cache.keys()):
            if (
                i in k[1:]
                or any(isinstance(arg, tuple) and i in arg for arg in k[1:])
                or k in [("joint",), ("joint_m",)]
            ):
                del self._cache[k]

    def drop_last_part(self):
        assert self.n_parts >= 2
        i = self.n_parts - 1
        self._invalidate_cache(i)
        del self._buffers[f"_part_{i}"]
        self.n_parts -= 1

    def change_part(self, i, new):
        assert i < self.n_parts
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._invalidate_cache(i)
        self._buffers[f"_part_{i}"] = new

    def append_part(self, new):
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._buffers[f"_part_{self.n_parts}"] = new
        self.n_parts += 1

    ############################################################################
    # PyTorch-related stuff

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def device(self):
        return self.X.device

    def as_tensors(self, *args, **kwargs):
        "Helper that makes everything a tensor with self.X's type."
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("dtype", self.dtype)
        return tuple(None if r is None else torch.as_tensor(r, **kwargs) for r in args)

    def _apply(self, fn):  # used in to(), cuda(), etc
        super()._apply(fn)
        for key, val in self._cache.items():
            if val is not None:
                self._cache[key] = fn(val)
        return self

    def __copy__(self):
        """
        Doesn't deep-copy the data tensors, but copies dictionaries so that
        change_part/etc don't affect the original.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        to_copy = {"_cache", "_buffers", "_parameters", "_modules"}
        for k, v in self.__dict__.items():
            result.__dict__[k] = v.copy() if k in to_copy else v
        return result


class LazyKernelOnVectors(LazyKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.X.shape) != 2:
            raise ValueError(f"{type(self).__name__} expects 2d input tensors")


################################################################################
# Finally, the actual kernel functions.


@register
class Linear(LazyKernelOnVectors):
    r"k(x, y) = <x, y>"

    def _compute(self, A, B):
        return A @ B.t()


@register
class Polynomial(LazyKernelOnVectors):
    r"""
    k(x, y) = (gamma <x, y> + coef0)^degree

    gamma=None (the default) means to use 1/dimension
    """

    def __init__(
        self, X, *rest, degree: float = 3, gamma: float_or_none = None, coef0: float = 1
    ):
        super().__init__(X, *rest)
        self.degree = degree
        self.gamma = 1 / X.shape[1] if gamma is None else gamma
        self.coef0 = coef0

    def _compute(self, A, B):
        XY = A @ B.t()
        return (self.gamma * XY + self.coef0) ** self.degree


@register
class LinearAndSquare(LazyKernelOnVectors):
    r"k(x, y) = <x, y> + w <x^2, y^2>, with the squaring elementwise."

    def __init__(self, *parts, w: float = 1):
        super().__init__(*parts)
        self.w = w

    def _precompute(self, A):
        return (A * A,)

    def _compute(self, A, A_squared, B, B_squared):
        return A @ A.t() + self.w * (A_squared @ B_squared.t())


@register
class MixRBFDot(LazyKernelOnVectors):
    r"""
    k(x, y) = \sum_i wts[i] exp(- ||x - y||^2 / (2 * sigmas_sq[i])) + add_dot * <x, y>

    wts=None (the default) uses 1/len(sigmas_sq) for each weight.
    """

    def __init__(
        self,
        *parts,
        sigmas_sq: floats = (1,),
        wts: floats_or_none = None,
        add_dot: float = 0,
    ):
        super().__init__(*parts)
        self.sigmas_sq, self.wts = self.as_tensors(sigmas_sq, wts)
        if wts is not None:
            assert self.sigmas_sq.shape == self.wts.shape
        self.add_dot = add_dot
        if not add_dot:
            self.const_diagonal = 1 if self.wts is None else self.wts.sum()

    def _precompute(self, A):
        return (torch.einsum("ij,ij->i", A, A),)

    def _compute(self, A, A_sqnorms, B, B_sqnorms):
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
    r"""
    k(x, y) = \sum_i wts[i] exp(- ||x - y||^2 / (2 * sigmas_sq[i]))
    """

    def __init__(self, *parts, sigmas_sq: floats = (1,), wts: floats_or_none = None):
        super().__init__(*parts, sigmas_sq=sigmas_sq, wts=wts, add_dot=0)


@register
class RBF(MixRBFDot):
    r"""
    k(x, y) = exp(- ||x - y||^2 / (2 * sigma_sq))
    """

    def __init__(self, *parts, sigma_sq: float = 1):
        super().__init__(*parts, sigmas_sq=(sigma_sq,), wts=None, add_dot=0)


@register
class MixRQDot(LazyKernelOnVectors):
    r"""
    k(x, y) =
        \sum_i wts[i] (1 + ||x - y||^2 / (2 * alphas[i] * bws_sq[i]))^(-alphas[i])
        + add_dot * <x, y>
    """

    def __init__(
        self,
        *parts,
        alphas: floats = (1,),
        bws_sq: floats = (1,),
        wts: floats_or_none = None,
        add_dot: float = 0,
    ):
        super().__init__(*parts)
        self.alphas, self.bws_sq, self.wts = self.as_tensors(alphas, bws_sq, wts)
        assert self.alphas.shape == self.bws_sq.shape
        if wts is not None:
            assert self.sigmas_sq.shape == self.wts.shape
        self.add_dot = add_dot
        if not add_dot:
            self.const_diagonal = 1 if self.wts is None else self.wts.sum()

    def _precompute(self, A):
        return (torch.einsum("ij,ij->i", A, A),)

    def _compute(self, A, A_sqnorms, B, B_sqnorms):
        dot = A @ B.t()
        D2 = A_sqnorms[:, None] + B_sqnorms[None, :] - 2 * dot
        alphas = self.alphas[:, None, None]
        bws_sq = self.bws_sq[:, None, None]
        K_parts = (1 + D2[None, :, :] / (2 * alphas * bws_sq)) ** (-alphas)
        if self.wts is None:
            K = K_parts.mean(0)
        else:
            K = torch.einsum("sij,s->ij", K_parts, self.wts)
        return (K + self.add_dot * dot) if self.add_dot else K


@register
class MixRQ(MixRQDot):
    r"""
    k(x, y) =
        \sum_i wts[i] (1 + ||x - y||^2 / (2 * alphas[i] * bws_sq[i]))^(-alphas[i])
    """

    def __init__(
        self,
        *parts,
        alphas: floats = (1,),
        bws_sq: floats = (1,),
        wts: floats_or_none = None,
    ):
        super().__init__(*parts, alphas=alphas, bws_sq=bws_sq, wts=wts, add_dot=0)


@register
class RQ(MixRQDot):
    r"k(x, y) = (1 + ||x - y||^2 / (2 * alpha * bw_sq))^(-alpha)"

    def __init__(self, *parts, alpha: float = 1, bw_sq: float = 1):
        super().__init__(*parts, alphas=(alpha,), bws_sq=(bw_sq,), wts=None, add_dot=0)
