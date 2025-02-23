from sparsematrix import SparseMatrix
from sparsequantumgates import SparseQuantumGates
from quantumregister import QuantumRegister
import time
import numpy as np

def grover_search(n, marked_state):
    """
    Implement Grover's search algorithm
    
    Parameters:
    ------------
        n_qubits: Number of qubits
        marked_state: Integer representing the marked state
    
    Returns:
    ------------
        register: Final quantum register after applying Grover's algorithm
    """
    
    # Create quantum register
    register = QuantumRegister(n)
    # Calculate number of Grover iterations
    size = 2 ** n
    iterations = int((np.pi / 4) * (size ** 0.5))

    gates = SparseQuantumGates(n)
    
    # Create operators
    oracle = gates.Oracle(marked_state)
    diffusion = gates.Diffusion()
    
    # Apply Grover iterations
    for _ in range(iterations):
        # Apply oracle
        register.apply_gate(oracle)
        # Apply diffusion
        register.apply_gate(diffusion)
    
    result = register.measure()
    return result


#Test the Algorithm

if __name__ == "__main__":
    n_qubits = 6  # Number of qubits
    marked_state = 24  # State we're searching for
    

    start_time = time.time()
    result = grover_search(n_qubits, marked_state)
    end_time = time.time()
    print(f"Result: {result}")
    print(f"Time taken: {end_time - start_time} seconds")
    
        
