from sparsematrix import SparseMatrix
import numpy as np

class QuantumRegister:
    def __init__(self, n):
        """
        Initialize a quantum register with n qubits in equal superposition

        Parameters:
        ------------
            n: Number of qubits
        """
        self.n = n
        self.size = 2 ** n
        self.state = SparseMatrix(self.size, 1)
        initial_amplitude = 1.0 / np.sqrt(self.size)
        
        # Set initial superposition state
        for i in range(self.size):
            self.state[i, 0] = initial_amplitude
    
    def apply_gate(self, gate):
        """
        Apply a quantum gate (sparse matrix) to the quantum state

        Parameters:
        ------------
            gate: SparseMatrix representing the quantum gate
        """
        self.state = gate * self.state
    
    def measure(self):
        """
        Measure the quantum state and return a state based on probabilitiess
        """
        probabilities = {}
        total_prob = 0

        for (row, col), amplitude in self.state.data.items():
            prob = abs(amplitude) ** 2
            probabilities[row] = prob
            total_prob += prob

        # Normalize probabilities
        for key in probabilities:
            probabilities[key] /= total_prob

        # Sample based on probabilities
        return np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
