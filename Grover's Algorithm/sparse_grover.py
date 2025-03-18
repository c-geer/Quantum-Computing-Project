from sparsematrix import SparseMatrix
import random
import numpy as np
import time


# Sparse Hadamard gate
def hadamard_gate():
    return SparseMatrix.from_dense_matrix([[1 / np.sqrt(2), 1 / np.sqrt(2)], 
                                            [1 / np.sqrt(2), -1 / np.sqrt(2)]])

# Sparse Pauli-X gate
def pauli_x_gate():
    return SparseMatrix.from_dense_matrix([[0, 1], 
                                           [1, 0]])

def apply_gate_sparse(state, gate, qubit, n):
    """Applies a single-qubit gate to the given qubit in an n-qubit system using sparse representation."""
    full_gate = SparseMatrix(1, 1).from_dense([[1]])  # Start with identity
    identity = SparseMatrix.from_dense_matrix([[1, 0], [0, 1]])

    # Construct the full matrix as a tensor product of identities and the gate
        
    for i in range(n):
        full_gate = full_gate.tensor_product(gate if i == qubit else identity)  # Identity

    return full_gate * state  # Perform sparse matrix-vector multiplication

def multi_controlled_z_sparse(n):
    """Constructs a multi-controlled Z gate as a sparse matrix."""
    size = 2**n
    sparse_Z = SparseMatrix(size, size)
    for i in range(size):
        sparse_Z[i, i] = -1 if i == size - 1 else 1  # Flip only |111...1>

    return sparse_Z

def grover_oracle_sparse(state, marked_index, n):
    """Flips the amplitude of the marked state using sparse representation."""
    oracle = SparseMatrix(2**n, 2**n)
    for i in range(2**n):
        oracle[i, i] = -1 if i == marked_index else 1

    return oracle * state

def diffusion_operator_sparse(state, n):
    """Applies Grover's diffusion operator using sparse matrices."""
    hadamard = hadamard_gate()
    pauli_x = pauli_x_gate()

    for qubit in range(n):
        state = apply_gate_sparse(state, hadamard, qubit, n)
    
    for qubit in range(n):
        state = apply_gate_sparse(state, pauli_x, qubit, n)

    state = multi_controlled_z_sparse(n) * state  # Apply MC-Z as a sparse matrix

    for qubit in range(n):
        state = apply_gate_sparse(state, pauli_x, qubit, n)
    
    for qubit in range(n):
        state = apply_gate_sparse(state, hadamard, qubit, n)

    return state


def grovers_algorithm_sparse(n, marked_index):
    """Runs Grover's algorithm with sparse matrices."""
    size = 2**n
    state = SparseMatrix(size, 1)
    state[0, 0] = 1  # Initial state |0...0>

    h = hadamard_gate()
    # Apply Hadamard transform to all qubits
    for qubit in range(n):
        state = apply_gate_sparse(state, h, qubit, n)

    iterations = int(np.pi / 4 * size**0.5)

    for _ in range(iterations):
        state = grover_oracle_sparse(state, marked_index, n)
        state = diffusion_operator_sparse(state, n)

    return state

"""
n = 4
marked_index = 5
start_time = time.time()
result = grovers_algorithm_sparse(n, marked_index)
end_time = time.time()
result = result.to_dense()
probs = np.abs(result)**2
most_likely = random.choices(range(len(probs)), probs)[0]
print(f"Most likely state: {most_likely} (binary: {format(most_likely, f'0{n}b')})")
print(f"Time taken for {n} qubits: {end_time-start_time}")
"""

