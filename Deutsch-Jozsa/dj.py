import numpy as np
from gates import h_gate
from lazy import LazyCircuit
from tensor import Tensor

def measure_n(state):
    # perform some operations
    return measurement

def oracle(args):
    # this one's all you, Oskar
    return oracle_matrix

def deutsch_jozsa(n, f):
    size = 2**n # account for ancilla quibit
    v = np.zeros(size, dtype=np.complex128)

    return False
    
    

if __name__ == "__main__":
    n = 3
    func = ("constant", "balanced")

    is_constant = deutsch_jozsa(n, func[0])
    
    if is_constant:
        print("The function is constant")
    else:
        print("The function is balanced")
    