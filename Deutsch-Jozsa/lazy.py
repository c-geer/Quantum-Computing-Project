import numpy as np

class QuantumGate:
    def __init__(self, squarematrix, qbpos):
        self.sm = np.array(squarematrix)
        self.smDim = self.sm.shape[0]
        self.qbpos = qbpos
        self.lazy_ops = []

    def gather(self, i):
        j = 0
        for k, pos in enumerate(self.qbpos):
            j |= ((i >> pos) & 1) << k
        return j

    def scatter(self, j):
        i = 0
        for k, pos in enumerate(self.qbpos):
            i |= ((j >> k) & 1) << pos
        return i

    def lazy_apply(self, v):
        self.lazy_ops.append(v)

    def compute(self):
        v = np.array([1] + [0] * (2**len(self.qbpos) - 1), dtype=np.complex128)  # Initialize state |0...0>
        for vec in self.lazy_ops:
            v = self._apply(v)
        return v

    def _apply(self, v):
        v = np.array(v)
        w = np.zeros_like(v, dtype=np.complex128)
        for i in range(len(v)):
            r = self.gather(i)
            for c in range(self.smDim):
                j = (i & ~self.scatter(self.smDim-1)) | self.scatter(c)
                w[i] += self.sm[r, c] * v[j]
        return w

def hadamard_gate(n):
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    H_n = H
    for _ in range(n-1):
        H_n = np.kron(H_n, H)
    return H_n

# Example usage
if __name__ == "__main__":
    n = 3  # Number of qubits
    H_n = hadamard_gate(n)
    
    # Create QuantumGate object
    gate = QuantumGate(H_n, list(range(n)))

    # Add lazy operation
    gate.lazy_apply(None)  # State vector is handled internally

    # Compute the result
    result = gate.compute()
    print(f"Computed Result:\n{result}")