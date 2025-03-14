import numpy as np
from gates import h_gate
from gates import x_gate
from gates import cx_gate
from lazy import LazyCircuit
from tensor import Tensor

def measure_n(n, v):
    # perform measurement
    return measurement

def oracle(n,f):
    # this one's all you, Oskar

    if f == "constant":
        #the constant case 
        #the x gate has to be the size of 1 Q bit not n big 
        oracle_matrix = x_gate(1)      
        
        qblist = list(range(n-1,n))
        
        return oracle_matrix, qblist
        
    else:
        #the balanced case 
        oracle_matrix = cx_gate(n)
        
        qblist= list(range(n))
        
        return oracle_matrix, qblist
            
    return oracle_matrix

def remove_ancilla(n, v):
    # Reshape state vector to separate the ancilla qubit
    reshaped_state = v.reshape([2] * (n+1))

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
    circuit1 = LazyCircuit()

    # step 1
    H = h_gate(n+1)
    circuit1.lazy_apply(H, list(range(n+1)))

    # step 2
    U, qblist = oracle(n, f)
    circuit1.lazy_apply(U, qblist)

    #step 3 
    H_n = h_gate(n)
    circuit1.lazy_apply(H_n, list(range(1, n)))

    #step 4 compute the state of the register 
    state = circuit1.compute(v[0])
    print(state)

    #step 5 remove the ancilla 
    final_state = remove_ancilla(n, state)
    
    
    #step 6 measure 
    #take the absolure value of the amplitudes 
    measured_amplitude = np.abs(final_state[0][0])

    if np.isclose(1,measured_amplitude , rtol = 0.1):
        return True 
        
    else:
    
        return False
    
    

if __name__ == "__main__":
    n = 3
    func = ("constant", "balanced")
    is_constant = True 

    is_constant = deutsch_jozsa(n, func[1])
    
    if is_constant:
        print("The function is constant")
    else:
        print("The function is balanced")
    
