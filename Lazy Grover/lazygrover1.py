import numpy as np
import time
import random
from tensor import Tensor
from lazy import LazyCircuit
from gates import h_gate
from gates import x_gate
from gates import multi_cz_gate
from gates import grovers_oracle

def grovers_algo(n, marked_index):
    H_n = h_gate(n)
    X_n = x_gate(n)
    MCZ = multi_cz_gate(n)
    oracle = grovers_oracle(n, marked_index)
    size = 2**n
    iterations = int(np.pi / 4 * size**0.5)

    gates = LazyCircuit(list(range(n)))

    gates.lazy_apply(H_n)

    for _ in range(iterations):
        gates.lazy_apply(oracle)
        gates.lazy_apply(H_n)
        gates.lazy_apply(X_n)
        gates.lazy_apply(MCZ)
        gates.lazy_apply(X_n)
        gates.lazy_apply(H_n)

    state = Tensor(np.zeros(2**n, dtype=np.complex128))  # Initialize state
    state[0][0] = 1
    result = gates.compute(state[0])
    return result
"""
if __name__ == "__main__":
    n = 4  # Number of qubits
    marked_index = 7
    result = grovers_algo(n, marked_index)
    probs = np.abs(result)**2
    index = random.choices(range(len(probs)), probs)[0]
    print(f"Computed Result:\n{result}")
    print(f"Determined Index: {index}")""
"""

