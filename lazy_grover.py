import numpy as np
import time

def apply_hadamard_lazy(state, qubit, n):
    """Applies Hadamard gate lazily to a specific qubit."""
    size = len(state)
    new_state = state.copy()

    for i in range(size):
        if (i >> (n - 1 - qubit)) & 1:  # If qubit is |1⟩
            i0 = i & ~(1 << (n - 1 - qubit))  # Flip to |0⟩ state
            new_state[i] = (state[i0] - state[i]) / np.sqrt(2)
        else:  # If qubit is |0⟩
            i1 = i | (1 << (n - 1 - qubit))  # Flip to |1⟩ state
            new_state[i] = (state[i] + state[i1]) / np.sqrt(2)

    return new_state

def apply_pauli_x_lazy(state, qubit, n):
    """Applies Pauli-X (bit flip) lazily."""
    size = len(state)
    new_state = state.copy()

    for i in range(size):
        if (i >> (n - 1 - qubit)) & 1 == 0:  # If qubit is |0⟩
            i1 = i | (1 << (n - 1 - qubit))  # Flip to |1⟩
            new_state[i], new_state[i1] = new_state[i1], new_state[i]  # Swap

    return new_state

def diffusion_operator_lazy(state, n):
    """Applies Grover's diffusion operator lazily."""
    for qubit in range(n):
        state = apply_hadamard_lazy(state, qubit, n)
    
    for qubit in range(n):
        state = apply_pauli_x_lazy(state, qubit, n)

    # Flip phase of |111...1>
    target_index = (1 << n) - 1
    state[target_index] *= -1  

    for qubit in range(n):
        state = apply_pauli_x_lazy(state, qubit, n)

    for qubit in range(n):
        state = apply_hadamard_lazy(state, qubit, n)
    
    return state

def grovers_algorithm_lazy(n, marked_index):
    """Runs Grover's algorithm using lazy gate applications."""
    size = 2**n
    state = np.zeros(size)
    state[0] = 1
    for qubit in range(n):
        state = apply_hadamard_lazy(state, qubit, n)  # Uniform superposition

    iterations = int(np.pi / 4 * np.sqrt(size))

    for _ in range(iterations):
        # Oracle: Flip amplitude of marked state
        state[marked_index] *= -1

        # Apply lazy diffusion operator
        state = diffusion_operator_lazy(state, n)

    return state

n = 12
marked_index = 5

start_time = time.time()
state = grovers_algorithm_lazy(n, marked_index)
end_time = time.time()
print(state)
most_likely = np.argmax(np.abs(state))  
print(f"Most likely state: {most_likely} (binary: {format(most_likely, f'0{n}b')})")
print(f"Time taken: {end_time-start_time}")

