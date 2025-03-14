import numpy as np
from gates import h_gate
from gates import x_gate
from gates import cx_gate
from lazy import LazyCircuit
from tensor import Tensor
import random

def measure_n(n, v):
    # Calculate probabilities for each possible outcome
    probabilities = np.abs(v)**2
    
    # Initialize array to hold measurement results for the first n qubits
    measurement_results = np.zeros(2**n, dtype=float)
    
    for i in range(2**(n + 1)):
        # Extract the first n bits of the index
        index = i >> 1
        # Add the probability to the corresponding measurement result
        measurement_results[index] += probabilities[i]

    probs = np.abs(measurement_results)**2
    measurement = random.choices(range(len(measurement_results)), measurement_results)[0]
    print(measurement)

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
    
    #step 5 measure first n qubits
    measurement = measure_n(n, state)

    if measurement == 0:
        return True 
    else:
        return False
    
    

if __name__ == "__main__":
    n = 3
    func = ("constant", "balanced")
    is_constant = True 

    is_constant = deutsch_jozsa(n, func[0])

    if is_constant:
        print("The function is constant")
    else:
        print("The function is balanced")
    
