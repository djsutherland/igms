import torch


def mix_rbf_dot(X, Y=None, sigmas_sq=(1,), wts=None, add_dot=0, self_Ks=True):
    X = torch.as_tensor(X)
    if Y is not None:
        Y = torch.as_tensor(Y, dtype=X.dtype, device=X.device)
    if wts is not None:
        wts = torch.as_tensor(wts, dtype=X.dtype, device=X.device)
    sigmas_sq = torch.as_tensor(sigmas_sq, dtype=X.dtype, device=X.device)

    XX = X @ X.t()
    X_sqnorms = torch.diagonal(XX)

    if Y is None:
        YY = XX
        Y_sqnorms = X_sqnorms
        XY = XX
    else:
        YY = Y @ Y.t()
        Y_sqnorms = torch.diagonal(YY)
        XY = X @ Y.t()

    sqdists = [(X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY, XY)]
    if self_Ks and Y is not None:
        sqdists.append((X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX, XX))
        sqdists.append((Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY, YY))

    Ks = []
    for D2, lin in sqdists:
        K_parts = torch.exp(D2[None, :, :] / (-2 * sigmas_sq[:, None, None]))
        if wts is None:
            K = K_parts.mean(0)
        else:
            K = torch.einsum("sij,s->ij", K_parts, wts)
        if add_dot:
            K = K + add_dot * lin
        Ks.append(K)

    if not self_Ks:
        return Ks[0]
    elif Y is None:
        K, = Ks
        return [K, K, K]
    else:
        return Ks


def linear(X, Y=None, self_Ks=True):
    X = torch.as_tensor(X)
    if Y is not None:
        Y = torch.as_tensor(Y, dtype=X.dtype, device=X.device)

    XY = X @ (X if Y is None else Y).t()

    if self_Ks:
        if Y is not None:
            XX = X @ X.t()
            YY = Y @ Y.t()
            return [XY, XX, YY]
        else:
            return [XY, XY, XY]
    else:
        return XY
