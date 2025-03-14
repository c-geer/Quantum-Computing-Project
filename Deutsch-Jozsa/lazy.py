import numpy as np
from gates import h_gate
from gates import cz_gate
from gates import multi_cz_gate
from gates import grovers_oracle
from gates import x_gate
from sparsematrix import SparseMatrix
from tensor import Tensor

class LazyCircuit:
    def __init__(self):
        """Initialize with qubit positions we want to act upon."""
        self.lazy_ops = []  # Store gates instead of state vectors
        self.qblist = [] # Store positions of qubits we want to apply each gate to

    def gather(self, i, qbpos):
        """Maps a global index to a local index for the subspace."""
        j = 0
        for k, pos in enumerate(qbpos):
            j |= ((i >> pos) & 1) << k
        return j

    def scatter(self, j, qbpos):
        """Maps a local index back to a global index."""
        i = 0
        for k, pos in enumerate(qbpos):
            i |= ((j >> k) & 1) << pos
        return i

    def lazy_apply(self, gate, qbpos):
        """Queue a gate operation to be applied lazily."""
        self.lazy_ops.append(gate)
        self.qblist.append(qbpos)

    def compute(self, v):
        """Apply all queued gates to the input state"""
        for i in range(len(self.lazy_ops)):
            v = self._apply(self.lazy_ops[i], v, self.qblist[i])  # Apply each gate in sequence
        return v

    def _apply(self, gate, v, qbpos):
        """Applies a given gate to the input state vector v."""
        v = np.array(v)
        w = np.zeros_like(v, dtype=np.complex128)

        for i in range(len(v)):
            r = self.gather(i, qbpos)  # Get corresponding local index
            for c in range(gate.shape[0]):
                j = (i & ~self.scatter(gate.shape[0] - 1, qbpos)) | self.scatter(c, qbpos)
                w[j] += gate[r, c] * v[i]  # Apply gate correctly

        return w


# Example usage
if __name__ == "__main__":
    n = 2  # Number of qubits
    H_n = h_gate(n)
    Z_n = cz_gate(n, 0, 2)
    mcz = multi_cz_gate(n)
    oracle = grovers_oracle(n, 3)
    X_n = x_gate(n)
    
    # Create LazyCircuit object
    circuit = LazyCircuit()

    # Queue your desired gate operations
    qblist = list(range(1, n))
    circuit.lazy_apply(X_n, qblist)
    #qblist = list(range(n))
    #gate.lazy_apply(H_n, qblist)

    # Compute the final state after applying all queued gates
    v = Tensor(np.zeros(2**n, dtype=np.complex128))  # Initialize state
    v[0][1] = 1
    print(f"Initial state:\n{v}")
    result = circuit.compute(v[0])
    print(f"Computed Result:\n{result}")
