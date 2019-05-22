import torch


def mmd2(K_XY, K_XX, K_YY, biased=False):
    K_XY = torch.as_tensor(K_XY)
    K_XX = torch.as_tensor(K_XX)
    K_YY = torch.as_tensor(K_YY)

    if biased:
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m, n = K_XY.shape
        return (
            (K_XX.sum() - K_XX.trace()) / (m * (m - 1))
            + (K_YY.sum() - K_YY.trace()) / (n * (n - 1))
            - 2 * K_XY.mean()
        )
