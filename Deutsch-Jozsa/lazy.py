import numpy as np
from gates import h_gate
from gates import cz_gate
from sparsematrix import SparseMatrix

class LazyCircuit:
    def __init__(self, qbpos):
        """Initialize with qubit positions we want to act upon."""
        self.qbpos = qbpos
        self.lazy_ops = []  # Store gates instead of state vectors

    def gather(self, i):
        """Maps a global index to a local index for the subspace."""
        j = 0
        for k, pos in enumerate(self.qbpos):
            j |= ((i >> pos) & 1) << k
        return j

    def scatter(self, j):
        """Maps a local index back to a global index."""
        i = 0
        for k, pos in enumerate(self.qbpos):
            i |= ((j >> k) & 1) << pos
        return i

    def lazy_apply(self, gate):
        """Queue a gate operation to be applied lazily."""
        self.lazy_ops.append(gate)

    def compute(self):
        """Apply all queued gates to the |0...0⟩ initial state."""
        num_qubits = len(self.qbpos)
        v = np.zeros(2**num_qubits, dtype=np.complex128)  # Initialize |0...0⟩ state
        v[0] = 1

        for gate in self.lazy_ops:
            v = self._apply(gate, v)  # Apply each gate in sequence

        return v

    def _apply(self, gate, v):
        """Applies a given gate to the input state vector v."""
        v = np.array(v)
        w = np.zeros_like(v, dtype=np.complex128)

        for i in range(len(v)):
            r = self.gather(i)  # Get corresponding local index
            for c in range(gate.shape[0]):
                j = (i & ~self.scatter(gate.shape[0] - 1)) | self.scatter(c)
                w[j] += gate[r, c] * v[i]  # Apply gate correctly

        return w


# Example usage
if __name__ == "__main__":
    n = 3  # Number of qubits
    H_n = h_gate(n)
    Z_n = cz_gate(n, 0, 2)
    
    # Create LazyCircuit object
    gate = LazyCircuit(list(range(n)))

    # Queue the Hadamard operation twice
    gate.lazy_apply(H_n)
    gate.lazy_apply(Z_n) 
    # Scream in joy as you realise the code works

    # Compute the final state after applying all queued gates
    result = gate.compute()
    print(f"Computed Result:\n{result}")
