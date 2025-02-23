from sparsematrix import SparseMatrix

class SparseQuantumGates:
    def __init__(self, n):
        """
        Initialize the quantum gates with n qubits
        
        Parameters:
        ------------
            n: Number of qubits
        """
        self.n = n
        self.size = 2 ** n

    def Hadamard(self):
        """
        Create n-qubit Hadamard gate
        
        Returns:
        ------------
            SparseMatrix: The Hadamard gate
        """
        h = SparseMatrix(self.size, self.size)
        factor = 1 / (2 ** (self.n/2))
        
        for i in range(self.size):
            for j in range(self.size):
                # Calculate the dot product of binary representations
                dot_product = bin(i & j).count('1')
                h[i, j] = factor * (-1) ** dot_product
        
        return h

    def Oracle(self,marked_state):
        """
        Create oracle matrix for n qubits with given marked state
        
        Parameters:
        ------------
            marked_state: Integer representing the marked state

        Returns:
        ------------
            SparseMatrix: The oracle matrix
        """
        oracle = SparseMatrix(self.size, self.size)
        
        # Set diagonal elements to 1, except for marked state
        for i in range(self.size):
            oracle[i, i] = 1
        
        # Flip sign for marked state
        oracle[marked_state, marked_state] = -1
        
        return oracle

    def Diffusion(self):
        """
        Create diffusion operator for n qubits
        
        Returns:
        ------------
            SparseMatrix: The diffusion operator
        """
        diff = SparseMatrix(self.size, self.size)
        
        # Create matrix with 2/N - 1 on diagonal and 2/N elsewhere
        factor = 2.0 / self.size
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    diff[i, j] = factor - 1
                else:
                    diff[i, j] = factor
        
        return diff