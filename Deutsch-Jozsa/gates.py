import numpy as np
from scipy.sparse import identity
from tensor import Tensor

def x_gate(n):
    """X gate"""
    X = Tensor(np.array([[0, 1], [1, 0]], dtype=complex))
    X_n = X
    for _ in range(n-1):
        X_n = X_n.TensorProduct(X)
    return X_n

def h_gate(n):
    """Hadamard gate."""
    H = Tensor((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))
    H_n = H
    for _ in range(n-1):
        H_n = H_n.TensorProduct(H)
    return H_n

def z_gate(n):
    """Pauli-Z gate."""
    Z = Tensor(np.array([[1, 0], [0, -1]], dtype=complex))
    Z_n = Z
    for _ in range(n-1):
        Z_n = Z_n.TensorProduct(Z)
    return Z_n

def controlled_z(n, control, target):
    """Constructs a Controlled-Z gate for n qubits."""
    size = 2**n
    CZ = identity(size, dtype=complex)
    for i in range(size):
        if ((i >> control) & 1) and ((i >> target) & 1):  # Both control and target are |1⟩
            CZ[i, i] = -1
    return Tensor(CZ)
