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
        w = np.zeros_like(v, dtype=np.complex128)  

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
    def __init__(self, qbpos,n): #tensor product n times for however many qubits (all gates)
        h = Tensor(np.array([[1, 1], [1, -1]]))
        q = qbpos
        qbpos = qbpos[0]
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        if qbpos == 0:
            hadamard_ = h.TensorProduct(identity)
            for i in range(n-2):
                hadamard_ = hadamard_.TensorProduct(identity)
        else:
            hadamard_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    hadamard_ = hadamard_.TensorProduct(h)
                else:
                    hadamard_ = hadamard_.TensorProduct(identity)
        hadamard = np.array(hadamard_.data) * (1/np.sqrt(2))**(n)
        super().__init__(hadamard, q)

class CNOT(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the CNOT Gate class
        """
        cnot = Tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
        cnot_ = cnot
        for i in range(n-2):
            cnot_ = cnot_.TensorProduct(cnot)
        cnot = np.array(cnot_.data)
        
        super().__init__(cnot, qbpos)

class PhaseGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the Phase Gate class
        """
        phase = Tensor(np.array([[1, 0], [0, 1j]]))
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        q = qbpos
        qbpos = qbpos[0]
        if qbpos == 0:
            phase_ = phase.TensorProduct(identity)
            for i in range(n-2):
                phase_ = phase_.TensorProduct(identity)
        else:
            phase_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    phase_ = phase_.TensorProduct(phase)
                else:
                    phase_ = phase_.TensorProduct(identity)
                    
        phase = np.array(phase_.data)
        super().__init__(phase, q)

class ControlledVGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the Controlled-V Gate class
        """
        v = Tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]))
        v_ = v
        for i in range(n-2):
            v_ = v_.TensorProduct(v)
        v = np.array(v_.data)
        super().__init__(v, qbpos)

class InversePhaseGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the Inverse Phase Gate class
        """
        q = qbpos
        inversephase = Tensor(np.array([[1, 0], [0, -1j]]))
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        qbpos = qbpos[0]
        
        if qbpos == 0:
            inversephase_ = inversephase.TensorProduct(identity)
            for i in range(n-2):
                inversephase_ = inversephase_.TensorProduct(identity)
        else:
            inversephase_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    inversephase_ = inversephase_.TensorProduct(inversephase)
                else:
                    inversephase_ = inversephase_.TensorProduct(identity)
                    
        inversephase = np.array(inversephase_.data)
        
        super().__init__(inversephase, q)

class TGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the T Gate class
        """
        q = qbpos
        t = Tensor(np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]))
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        qbpos = qbpos[0]
        if qbpos == 0:
            t_ = t.TensorProduct(identity)
            for i in range(n-2):
                t_ = t_.TensorProduct(identity)
        else:
            t_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    t_ = t_.TensorProduct(t)
                else:
                    t_ = t_.TensorProduct(identity)
                    
        t = np.array(t_.data)
        super().__init__(t, q)

class InverseTGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the inverse T Gate class
        """
        q = qbpos
        qbpos = qbpos[0]
        tdagger = Tensor(np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]]))
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        
        if qbpos == 0:
            tdagger_ = tdagger.TensorProduct(identity)
            for i in range(n-2):
                tdagger_ = tdagger_.TensorProduct(identity)
        else:
            tdagger_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    tdagger_ = tdagger_.TensorProduct(tdagger)
                else:
                    tdagger_ = tdagger_.TensorProduct(identity)
        tdagger = np.array(tdagger_.data)
        
        
        super().__init__(tdagger, q)

class PauliXGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the Pauli-x Gate class
        """
        q = qbpos
        qbpos = qbpos[0]
        x = Tensor(np.array([[0, 1], [1, 0]]))
        identity = Tensor(np.array([[1, 0], [0, 1]]))
        if qbpos == 0:
            x_ = x.TensorProduct(identity)
            for i in range(n-2):
                x_ = x_.TensorProduct(identity)
        else:
            x_ = identity
            for i in range(n-1):
                if i == qbpos-1:
                    x_ = x_.TensorProduct(x)
                else:
                    x_ = x_.TensorProduct(identity)
        x = np.array(x_.data)
        super().__init__(x, q)

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
    

class MCXGate(QuantumGate):
    def __init__(self, qbpos, n):
        """
        Constructor for the MCX Gate class
        """
        mcx = np.zeros((2**n, 2**n))
        for i in range(2**n):
            if i == (2**n)-2:
                mcx[i, i+1] = 1
            elif i == (2**n)-1:
                mcx[i, i-1] = 1
            else:
                mcx[i, i] = 1
        super().__init__(mcx, qbpos)
