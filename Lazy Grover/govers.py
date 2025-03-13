import numpy as np
import time
import matplotlib.pyplot as plt

# Basic matrix operations
def apply_gate(state, gate, qubit, n):
    """Applies a single-qubit gate to the given qubit in an n-qubit system."""
    size = 2**n
    new_state = np.zeros(size, dtype=complex)
    for i in range(size):
        binary = list(format(i, f'0{n}b'))  # Convert index to binary
        i0 = i & ~(1 << (n - 1 - qubit))  # Index if qubit is |0>
        i1 = i | (1 << (n - 1 - qubit))  # Index if qubit is |1>

        if binary[qubit] == '0':
            new_state[i] += gate[0][0] * state[i0] + gate[0][1] * state[i1]
        else:
            new_state[i] += gate[1][0] * state[i0] + gate[1][1] * state[i1]
    
    return new_state

def hadamard_gate():
    """Hadamard gate."""
    return np.array([[1 / 2**0.5, 1 / 2**0.5], [1 / 2**0.5, -1 / 2**0.5]])

def pauli_x_gate():
    """Pauli-X gate."""
    return np.array([[0, 1], [1, 0]])


def multi_controlled_z(state, n):
    """Applies a multi-controlled Z gate to flip the phase of the |111...1> state."""
    size = 2**n
    new_state = state[:]
    target_index = size - 1 
    for i in range(size):
        if i == target_index:  # Only flip phase of the target
            new_state[i] = -new_state[i]
    return new_state

# Grover's oracle 
def grover_oracle(state, marked_index):
    """Flips the amplitude of the marked state."""
    new_state = state[:]
    new_state[marked_index] *= -1
    return new_state

# Grover's diffusion operator
def diffusion_operator(state, n):
    """Applies the diffusion operator using fundamental gates."""
    # Step 1: Apply Hadamard to all qubits
    for qubit in range(n):
        state = apply_gate(state, hadamard_gate(), qubit, n)
    
    # Step 2: Apply Pauli-X to all qubits
    for qubit in range(n):
        state = apply_gate(state, pauli_x_gate(), qubit, n)
    
    # Step 3: Apply Multi-Controlled Z (phase flip)
    state = multi_controlled_z(state, n)
    
    # Step 4: Apply Pauli-X to all qubits again
    for qubit in range(n):
        state = apply_gate(state, pauli_x_gate(), qubit, n)
    
    # Step 5: Apply Hadamard to all qubits again
    for qubit in range(n):
        state = apply_gate(state, hadamard_gate(), qubit, n)
    
    return state

def grovers_algorithm(n, marked_index):
    """Runs Grover's algorithm on an n-qubit system with the marked state."""
    # Initialize state in equal superposition
    size = 2**n
    state = np.zeros(size)
    state[0] = 1
    for qubit in range(n):
        state = apply_gate(state, hadamard_gate(), qubit, n)

    # Grover's algorithm steps
    iterations = int(np.pi / 4 * size**0.5)  # Optimal number of iterations
    for _ in range(iterations):
        # Apply Oracle
        state = grover_oracle(state, marked_index)

        # Apply Diffusion Operator
        state = diffusion_operator(state, n)

    return state

"""
# Run Grover's algorithm 
n_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14])
time_list = []
for n in n_list:
    marked_index = 2**n - n
    print(f"Marked Index: {marked_index}")
    start_time = time.time()
    state = grovers_algorithm(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list.append(total_time)
    print(state)
    most_likely = np.argmax(np.abs(state))  
    print(f"Most likely state: {most_likely} (binary: {format(most_likely, f'0{n}b')})")
    print(f"Time taken: {total_time}")

time_list = np.array(time_list)
plt.plot(n_list, time_list)
plt.xlabel("Number of Qubits")
plt.ylabel("Time Taken in Seconds")
plt.title("Time Taken for Bit-Wise Lazy Implementation")
plt.show()
"""
# hi guys
