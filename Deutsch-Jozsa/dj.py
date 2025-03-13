import numpy as np
from gates import h_gate
from lazy import LazyCircuit
from tensor import Tensor

def measure_n(n, v):
    # perform measurement
    return measurement

def oracle(args):
    # this one's all you, Oskar
    return oracle_matrix

def remove_ancilla(n, v):
    # Reshape state vector to separate the ancilla qubit
    reshaped_state = v[0].reshape([2] * (n+1))

    # Sum over the ancilla dimension to trace it out
    reduced_state = np.sum(reshaped_state, axis=n)

    # Flatten the reduced state to get the new state vector
    n_v = reduced_state.flatten()

    # Normalize the new state vector
    n_v /= np.linalg.norm(n_v)

    return Tensor(n_v)

def deutsch_jozsa(n, f):

    # initialise first n qubits
    size_u = 2**n 
    u = Tensor(np.zeros(size_u, dtype=np.complex128))
    u[0][0] = 1

    # initialise ancilla qubit
    size_a = 2
    a = Tensor(np.zeros(size_a, dtype=np.complex128))
    a[0][1] = 1

    # combine into single register
    v = u.TensorProduct(a)

    # perform first 
    circuit1 = LazyCircuit(list(range(n+1)))

    # step 1
    H = h_gate(n+1)
    circuit1.lazy_apply(H)

    # step 2
    U = oracle(args)
    circuit1.lazy_apply(U)

    

    return False
    
    

if __name__ == "__main__":
    n = 3
    func = ("constant", "balanced")

    is_constant = deutsch_jozsa(n, func[0])
    
    if is_constant:
        print("The function is constant")
    else:
        print("The function is balanced")
    