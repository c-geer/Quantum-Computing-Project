import numpy as np
import time

class QuantumRegister:
    def __init__(self, n):
        """
        Initialize a quantum register with n qubits in equal superposition

        Parameters:
        n (int): Number of qubits
        """
        self.n = n # number of qubits
        self.size = 2**n # size of list
        self.state = np.ones(self.size) / np.sqrt(self.size)  # normalised states

    def measure(self):
        """
        Measure the quantum state by sampling based on probabilities
        
        Returns:
        int: The index of the measured state
        """
        probabilities = np.abs(self.state) ** 2 # P = |a|^2
        return np.random.choice(self.size, p=probabilities)

class LazyQuantumGate:
    def __init__(self, n):
        """
        Initialize a lazy quantum gate system

        Parameters:
        n (int): Number of qubits
        """ 
        self.n = n

    def lazyhadamard(self, state):
        """
        Apply Hadamard gate lazily and normalize

        Parameters:
        state (np.array): The quantum state

        Returns:
        np.array: The new quantum state

        ----------
        1 << bit: binary representation of 1 is shifted left by bit positions
        eg.
        1 << 2 = 100, because, 001 << 2 = 100 (1 is 001, 1 shifted left by 2 positions)
        3 << 2 = 1100, because, 011 << 2 = 1100 (3 is 011, 3 shifted left by 2 positions)

        a ^ b: a XOR b
        eg.
        3 ^ 1 = 2, because, 011 ^ 001 = 010 (where the bits are different, the result is 1, otherwise 0)

        Putting it together, 
        i ^ (1 << bit) flips the bit of i at the position specified by bit
        Note that we count from right starting from 0
        bit = 011101
        pos = 543210

        eg.
        5 = 101

        5 ^ (1 << 1) = 111 = 7
        5 ^ (1 << 0) = 101 = 5
        5 ^ (1 << 2) = 001 = 1

        Proof:
        5 ^ (1 << 1) = 101 ^ (001 << 1) = 101 ^ 010 = 111
        5 ^ (1 << 0) = 101 ^ (001 << 0) = 101 ^ 001 = 100
        5 ^ (1 << 2) = 101 ^ (001 << 2) = 101 ^ 100 = 001
        -----------
        """
        new_state = np.zeros_like(state, dtype=np.complex128)
        norm_factor = 1 / np.sqrt(2)  # Hadamard scaling factor

        for i in range(len(state)):
            for bit in range(self.n):
                new_state[i] += norm_factor * state[i ^ (1 << bit)]  # Apply bitwise Hadamard

        return new_state / np.linalg.norm(new_state)  # Normalize

    def oracle(self, state, marked_state):
        """
        Apply the oracle
        
        Parameters:
        state (np.array): The quantum state
        marked_state (int): The marked state

        Returns:
        np.array: The new quantum state
        """
        state = state.copy()
        state[marked_state] *= -1  # Flip the amplitude of the marked state
        return state

    def diffusion(self, state):
        """
        Apply the diffusion operator

        Parameters:
        state (np.array): The quantum state

        Returns:
        np.array: The new quantum state
        """
        avg_amplitude = np.mean(state)  # Compute the mean amplitude
        new_state = 2 * avg_amplitude - state  # Reflect around mean

        return new_state / np.linalg.norm(new_state)  # Normalize

class LazyGroverAlgorithm:
    def __init__(self, n, marked_state):
        """
        Initialize the Grover algorithm with n qubits and a marked state

        Parameters:
        n (int): Number of qubits
        marked_state (int): The marked state
        """
        self.n = n
        self.marked_state = marked_state
        self.qregister = QuantumRegister(n)
        self.gate = LazyQuantumGate(n)

        # Optimal number of iterations
        self.iterations = int(np.pi / 4 * np.sqrt(2**n))

    def apply_grover(self):
        """
        Apply the Grover algorithm
        
        Returns:
        int: The measured state
        """
        self.qregister.state = self.gate.lazyhadamard(self.qregister.state)  # Apply Hadamard

        for _ in range(self.iterations):
            self.qregister.state = self.gate.oracle(self.qregister.state, self.marked_state)  # Apply oracle
            self.qregister.state = self.gate.diffusion(self.qregister.state)  # Apply diffusion

        result = self.qregister.measure()  # Measure the final state
        return result

# test the algorithm
if __name__ == "__main__":
    n_qubits = 15    # Number of qubits
    marked_state = 10  # Marked state in binary: 1010

    start_time = time.time()
    grover = LazyGroverAlgorithm(n_qubits, marked_state)
    result = grover.apply_grover()
    end_time = time.time()

    print(f"Measured state: {bin(result)[2:].zfill(n_qubits)}")
    print(f"Time taken: {end_time - start_time} seconds")
