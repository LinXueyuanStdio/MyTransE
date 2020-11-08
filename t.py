import numpy as np
import torch


def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


a = torch.Tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
b = torch.Tensor([[2, 4, 6], [6, 8, 10], [10, 12, 14]])

M = torch.Tensor(rvs(3))
print(M)
c = torch.square(torch.norm(a - b, p=1, dim=1)).sum()
print(c)
d = torch.square(torch.norm(np.matmul(a, M) - b, p=1, dim=1)).sum()
print(d)
