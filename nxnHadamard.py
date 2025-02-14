import numpy as np

def hadamard_matrix(n):
    """
    Generate an n x n Hadamard matrix
    """
    if n == 1:
        return np.array([[1]])
    else:
        H_n_1 = hadamard_matrix(n // 2)
        return np.block([
            [H_n_1, H_n_1],
            [H_n_1, -H_n_1]
        ])