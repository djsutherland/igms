"""
Some probably over-engineered infrastructure for lazily computing kernel
matrices, allowing for various sums / means / etc used by MMD-related estimators.
"""
from functools import partial, wraps
import inspect

import numpy as np
import torch

from .utils import as_parameter, as_tensors, floats, float_or_none, floats_or_none


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
_bad_kinds = frozenset(
    (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
)


def register(original_cls=None, *, name=None):
    def decorator(cls):
        sig = inspect.signature(cls)
        arg_info = []
        for kw, param in sig.parameters.items():
            if param.kind in _bad_kinds:
                raise TypeError(
                    f"Can't register {cls}: argument {kw} has bad kind {param.kind}"
                )
            fn = param.annotation
            arg_info.append((kw, str if fn is inspect.Parameter.empty else fn))

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
# Kernel base class infrastructure


class Kernel(torch.nn.Module):
    # Note that _compute / _precompute functions should *not* modify
    # anything on self!

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

    def forward(self, *parts):
        """
        Compute the kernel on the given matrices. Usually you don't call this
        directly; instead use the generic pytorch __call__ method.

        Parts should be tensors representing objects you're computing the
        kernel among, so that eg a tensor of shape [n, 3] would represent
        n 3-dimensional objects.

        Returns a LazyKernelResult.
        """
        return LazyKernelResult(self, parts)

    def __repr__(self):
        return f"""<{type(self).__qualname__}({", ".join(
            f'''{n}={v.detach().cpu().item() if v.numel() == 1
                     else f'<shape {list(v.shape)}>'}'''
            for n, v in self._parameters.items()
        )})"""


_name_map = {"X": 0, "Y": 1, "Z": 2}


class LazyKernelResult:
    """
    Class that allows computing kernel matrices among a bunch of datasets,
    only computing the matrices when we use them.

    You normally won't construct this object directly, but instead
    get it from LazyKernel.__call__.

    Access the results with:
      - K[0, 1] to get the Tensor between parts 0 and 1.
      - K.XX, K.XY, K.ZY, etc: shortcuts, with X meaning 0, Y 1, Z 2.
      - K.matrix(0, 1) or K.XY_m: returns a Matrix subclass.
    """

    def __init__(self, kernel, parts):
        super().__init__()
        self.kernel = kernel
        self._cache = {}
        if not hasattr(self, "const_diagonal"):
            self.const_diagonal = False

        self._parts = as_tensors(*parts)
        shp = self.X.shape[1:]
        for p in self.parts[1:]:
            if p.shape is not None and p.shape[1:] != shp:
                raise ValueError(
                    f"inconsistent shapes: expected batch of {shp}, got {p.shape}"
                )

    def __repr__(self):
        return (
            f"<LazyKernelResult({self.kernel}, {', '.join(str(n) for n in self.ns)})>"
        )

    ############################################################################
    # Stuff that does the work of computing kernels

    def _compute(self, *args, **kwargs):
        return self.kernel._compute(*args, **kwargs)

    def _precompute(self, *args, **kwargs):
        return self.kernel._precompute(*args, **kwargs)

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

    # used to do this with __getattr__ but it's kind of annoying
    X = property(lambda self: self.part(0))
    Y = property(lambda self: self.part(1))
    Z = property(lambda self: self.part(2))

    XX = property(lambda self: self[0, 0])
    XY = property(lambda self: self[0, 1])
    XZ = property(lambda self: self[0, 2])
    YX = property(lambda self: self[1, 0])
    YY = property(lambda self: self[1, 1])
    YZ = property(lambda self: self[1, 2])
    ZX = property(lambda self: self[2, 0])
    ZY = property(lambda self: self[2, 1])
    ZZ = property(lambda self: self[2, 2])

    XX_m = property(lambda self: self.matrix(0, 0))
    XY_m = property(lambda self: self.matrix(0, 1))
    XZ_m = property(lambda self: self.matrix(0, 2))
    YX_m = property(lambda self: self.matrix(1, 0))
    YY_m = property(lambda self: self.matrix(1, 1))
    YZ_m = property(lambda self: self.matrix(1, 2))
    ZX_m = property(lambda self: self.matrix(2, 0))
    ZY_m = property(lambda self: self.matrix(2, 1))
    ZZ_m = property(lambda self: self.matrix(2, 2))

    n_parts = property(lambda self: len(self._parts))

    def _part(self, i):
        if i >= self.n_parts:
            raise AttributeError(f"have {self.n_parts} parts, asked for {i}")
        return self._parts[i]

    def part(self, i):
        p = self._part(i)
        return self.X if p is None else p

    @property
    def parts(self):
        return tuple(self.part(i) for i in range(self.n_parts))

    def n(self, i):
        return self.part(i).shape[0]

    @property
    def ns(self):
        return tuple(self.n(i) for i in range(self.n_parts))

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
        self._parts.pop(-1)

    def change_part(self, i, new):
        assert i < len(self._parts)
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._invalidate_cache(i)
        self._parts[i] = new

    def append_part(self, new):
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._parts.append(new)

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

    def __copy__(self):
        """
        Doesn't deep-copy the data tensors, but copies dictionaries so that
        change_part/etc don't affect the original.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        to_copy = {"_cache", "_parts"}
        for k, v in self.__dict__.items():
            result.__dict__[k] = v.copy() if k in to_copy else v
        return result


class KernelOnVectors(Kernel):
    def forward(self, *parts, **kwargs):
        parts = as_tensors(*parts)
        if any(len(p.shape) != 2 for p in parts):
            raise ValueError(f"{type(self).__name__} expects 2d input tensors")

        return super().forward(*parts, **kwargs)


################################################################################
# Finally, the actual kernel functions.


@register
class Linear(KernelOnVectors):
    r"k(x, y) = <x, y>"

    def _compute(self, A, B):
        return A @ B.t()


@register
class Polynomial(KernelOnVectors):
    r"""
    k(x, y) = (gamma <x, y> + coef0)^degree

    gamma=None (the default) means to use 1/dimension
    """

    def __init__(
        self, degree: float = 3.0, gamma: float_or_none = None, coef0: float = 1.0
    ):
        super().__init__()
        self.degree = as_parameter(degree)
        self._gamma = as_parameter(gamma)
        self.coef0 = as_parameter(coef0)

    def _compute(self, A, B):
        gamma = 1 / A.shape[1] if self._gamma is None else self._gamma
        XY = A @ B.t()
        return (gamma * XY + self.coef0) ** self.degree


@register
class LinearAndSquare(KernelOnVectors):
    r"k(x, y) = <x, y> + w <x^2, y^2>, with the squaring elementwise."

    def __init__(self, w: float = 1.0):
        super().__init__()
        self.w = as_parameter(w)

    def _precompute(self, A):
        return (A * A,)

    def _compute(self, A, A_squared, B, B_squared):
        return A @ B.t() + self.w * (A_squared @ B_squared.t())


class MixGeneralRBFDot(KernelOnVectors):
    """
    Base class for kernels which are a mixture of radial basis function
    kernels and a linear component.
    """

    def __init__(
        self,
        lengthscales_sq: floats = (1.0,),
        wts: floats_or_none = None,
        add_dot: float = 0.0,
    ):
        super().__init__()
        self.lengthscales_sq = as_parameter(lengthscales_sq)
        self.wts = as_parameter(wts)
        if wts is not None:
            assert self.lengthscales_sq.shape == self.wts.shape
        self.add_dot = as_parameter(add_dot)
        self._value_at_0 = 1

    @property
    def adding_dot(self):
        return self.add_dot.detach().cpu().item() != 0

    @property
    def const_diagonal(self):
        if self.addding_dot:
            return False
        elif self.wts is None:
            return self._value_at_0
        else:
            return self.wts.sum() * self._value_at_0

    def _precompute(self, A):
        return (torch.einsum("ij,ij->i", A, A),)

    def _compute(self, A, A_sqnorms, B, B_sqnorms):
        dot = A @ B.t()
        D2 = A_sqnorms[:, None] + B_sqnorms[None, :] - 2 * dot
        K_parts = self._rbf_function(
            D2[None, :, :] / self.lengthscales_sq[:, None, None]
        )
        if self.wts is None:
            K = K_parts.mean(0)
        else:
            K = torch.einsum("sij,s->ij", K_parts, self.wts)
        return (K + self.add_dot * dot) if self.adding_dot else K

    def _rbf_function(self, normalized_D2):
        raise NotImplementedError()


@register
class MixSqExpDot(MixGeneralRBFDot):
    r"""
    k(x, y) =
        \sum_i wts[i] exp(- ||x - y||^2 / (2 * lengthscales_sq[i]))
        + add_dot * <x, y>

    wts=None (the default) uses 1/len(lengthscales_sq) for each weight.
    """

    def _rbf_function(self, normalized_D2):
        return torch.exp(-0.5 * normalized_D2)


@register
class MixSqExp(MixSqExpDot):
    r"""
    k(x, y) = \sum_i wts[i] exp(- ||x - y||^2 / (2 * lengthscales_sq[i]))
    """

    def __init__(self, lengthscales_sq: floats = (1.0,), wts: floats_or_none = None):
        super().__init__(lengthscales_sq=lengthscales_sq, wts=wts, add_dot=0)


@register
class SqExp(MixSqExpDot):
    r"""
    k(x, y) = exp(- ||x - y||^2 / (2 * lengthscale_sq))
    """

    def __init__(self, lengthscale_sq: float = 1.0):
        super().__init__(lengthscales_sq=(lengthscale_sq,), wts=None, add_dot=0)


@register
class MixRQDot(MixGeneralRBFDot):
    r"""
    k(x, y) =
        \sum_i wts[i]
            (1 + ||x - y||^2 / (2 * alphas[i] * lengthscales_sq[i]))^(-alphas[i])
        + add_dot * <x, y>
    """

    def __init__(
        self,
        alphas: floats = (1.0,),
        lengthscales_sq: floats = (1.0,),
        wts: floats_or_none = None,
        add_dot: float = 0,
    ):
        super().__init__(lengthscales_sq=lengthscales_sq, wts=wts, add_dot=add_dot)

        self.alphas = as_parameter(alphas)
        assert self.alphas.shape == self.lengthscales_sq.shape

    def _rbf_function(self, normalized_D2):
        alphas = self.alphas[:, None, None]
        return (1 + normalized_D2 / (2 * alphas)) ** (-alphas)


@register
class MixRQ(MixRQDot):
    r"""
    k(x, y) =
        \sum_i wts[i]
            (1 + ||x - y||^2 / (2 * alphas[i] * lengthscales_sq[i]))^(-alphas[i])
    """

    def __init__(
        self,
        alphas: floats = (1.0,),
        lengthscales_sq: floats = (1.0,),
        wts: floats_or_none = None,
    ):
        super().__init__(
            alphas=alphas, lengthscales_sq=lengthscales_sq, wts=wts, add_dot=0
        )


@register
class RQ(MixRQDot):
    r"k(x, y) = (1 + ||x - y||^2 / (2 * alpha * lengthscale_sq))^(-alpha)"

    def __init__(self, alpha: float = 1.0, lengthscale_sq: float = 1.0):
        super().__init__(
            alphas=(alpha,), lengthscales_sq=(lengthscale_sq,), wts=None, add_dot=0
        )
