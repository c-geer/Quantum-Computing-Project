import numpy as np

def x_gate():
    """Pauli-X gate."""
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def h_gate():
    """Hadamard gate."""
    return (1 / np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex)

def z_gate():
    """Pauli-Z gate."""
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)

def controlled_z(n_qubits, control, target):
    """Constructs a controlled-Z gate for n qubits."""
    size = 2 ** n_qubits
    cz_matrix = np.eye(size, dtype=complex)
    for i in range(size):
        if ((i >> control) & 1) and ((i >> target) & 1):  # Both control and target are |1⟩
            cz_matrix[i, i] = -1
    return cz_matrix

def apply_single_qubit_gate(state, gate, n_qubits, qubit):
    """Applies a single-qubit gate to the specified qubit."""
    identity = np.eye(2, dtype=complex)
    gate_list = [identity] * n_qubits
    gate_list[qubit] = gate
    full_gate = gate_list[0]
    for g in gate_list[1:]:
        full_gate = np.kron(full_gate, g)
    return full_gate @ state

def grover_oracle_with_gates(n_qubits, marked_state_bits):
    """Constructs the oracle for a marked state using basic gates."""
    size = 2 ** n_qubits
    state = np.eye(size, dtype=complex)  # Identity matrix as the initial state

    # Step 1: Apply X gates to prepare the target state
    oracle_matrix = np.eye(size, dtype=complex)  # Start with an identity matrix
    for qubit, bit in enumerate(marked_state_bits):
        if bit == '0':
            x = apply_single_qubit_gate(state x).

