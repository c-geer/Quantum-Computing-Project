import numpy as np 
from tensor import Tensor

class QuantumGate(object):
    """Adaptation of C# Quantum gates from the lecture slides
    """
    
    def __init__(self, squarematrix, qbpos):
        """Constructor for the QuantumGate class

        Args:
            squarematrix (numpy array): any nxn matrix that represents the quantum gate 
            qbpos (int): positions of qubits that the gate acts on
        """
        self.sm = np.array(squarematrix)
        self.smDim = self.sm.shape[0]  # Dimension of matrix
        self.qbpos = qbpos
        
    def gather(self, i):
        """Extracts qubit values from an integer index i based on the qbpos array

        Args:
            i (int): integer index

        Returns:
            j (int): packed qubit index
        """
        j = 0
        for k, pos in enumerate(self.qbpos):
            j |= ((i >> pos) & 1) << k
        return j

    def scatter(self, j):
        """Converts a packed qubit index j back into its original position in the quantum state index i.

        Args:
            j (int): packed qubit index

        Returns:
            i (int): integer index with qubits in original positions
        """
        i = 0
        for k, pos in enumerate(self.qbpos):
            i |= ((j >> k) & 1) << pos
        return i

    def apply(self, v):
        """Applies the quantum gate to a quantum state vector v.

        Args:
            v (numpy array): quantum state vector

        Returns:
            w (array): output quantum state vector
        """
        v = np.array(v)  
        w = np.zeros_like(v)  

        for i in range(len(v)):
            r = self.gather(i)  
            i0 = i & ~self.scatter(r)  
            for c in range(self.smDim):
                j = i0 | self.scatter(c)  
                w[i] += self.sm[r, c] * v[j]  

        return w
    
def diracnotation(state):
    """Formats quantum state output in Dirac notation.
    Might need a bit more work
    """
    basis_states = [f"|{bin(i)[2:].zfill(3)}⟩" for i in range(len(state))]
    output = []
    
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-10:  # Ignore very small numbers (numerical noise)
            output.append(f"{amp:.3f} {basis_states[i]}")
    
    return " + ".join(output)

class HadamardGate(QuantumGate):
    """
    Hadamard gate class for a 3-qubit system
    """
    def __init__(self, qbpos):
        h = np.array([[1, 1], [1, -1]])
        unitmatrix2d = np.array([[1, 0], [0, 1]])
        unittensor = Tensor(unitmatrix2d)
        hadamard = Tensor(h)
        
        h1h = hadamard.TensorProduct((unittensor.TensorProduct(hadamard)))
        h1h = np.array(h1h.data)
        
        super().__init__(h1h, qbpos)

class CNOT(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the CNOT Gate class
        """
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        super().__init__(cnot, qbpos)

class PhaseGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the Phase Gate class
        """
        phase = np.array([[1, 0], [0, 1j]])
        super().__init__(phase, qbpos)

class ControlledVGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the Controlled-V Gate class
        """
        v = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])
        super().__init__(v, qbpos)

class InversePhaseGate(QuantumGate):
     def __init__(self, qbpos):
        """
        Constructor for the Inverse Phase Gate class
        """
        inversephase = np.array([[1, 0], [0, -1j]])
        super().__init__(inversephase, qbpos)

class TGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the T Gate class
        """
        t = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
        super().__init__(t, qbpos)

class InverseTGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the inverse T Gate class
        """
        tdagger = np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]])
        super().__init__(tdagger, qbpos)

class PauliXGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the Pauli-x Gate class
        """
        x = np.array([[0, 1], [1, 0]])
        super().__init__(x, qbpos)

class ToffoliGate(QuantumGate):
    def __init__(self, qbpos):
        """
        Constructor for the Toffoli Gate class
        """
        toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]])
        super().__init__(toffoli, qbpos)
    

    

