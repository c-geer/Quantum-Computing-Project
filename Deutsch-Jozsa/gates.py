import numpy as np
from tensor import Tensor
from sparsematrix import SparseMatrix

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

def cz_gate(n, control, target):
    """Constructs a Controlled-Z gate for n qubits."""
    size = 2**n
    CZ = SparseMatrix.from_dense_matrix(np.eye(size, dtype=complex))
    for i in range(size):
        if ((i >> control) & 1) and ((i >> target) & 1):  # Both control and target are |1⟩
            CZ[i, i] = -1
    return CZ

def cx_gate(n):
    CX = Tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    CX_n = CX
    for _ in range(n-1):
        CX_n = CX_n.TensorProduct(CX)
    return CX_n 

def multi_cz_gate(n):
    size = 2**n
    MCZ = SparseMatrix.from_dense_matrix(np.eye(size, dtype=complex))
    target_index = (1 << n) - 1
    MCZ[target_index, target_index] *= -1
    return MCZ

def grovers_oracle(n, marked):
    size = 2**n
    oracle = SparseMatrix.from_dense_matrix(np.eye(size, dtype=complex))
    oracle[marked,marked] *= -1
    return oracle

def CNOT_gate(n):
    #function that returns a C not matrix in tensot product form 
    C_n = Tensor(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128))
    return C_n
    
def dj_oracle(n, f_map):
    """Generate an oracle matrix based on the given function mapping."""
    
    num_qubits = n + 1
    U = SparseMatrix.from_dense_matrix(np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)) # Start with a matrix of zeroes.
    
    # Quantum state looks like IN-IN-...-IN-ANCILLA
    for input_state in range(2**num_qubits): # For each possible input
        input_string = input_state >> 1 # remove ANCILLA
        output_qubit = (input_state & 1) ^ (f_map[input_string]) # remove IN, XOR with f(IN)
        output_state = (input_string << 1) + output_qubit # the full state, with new OUT
        U[input_state, output_state] = 1 # set that part of U to 1

    return U

    

