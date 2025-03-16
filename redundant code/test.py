from tensor import Tensor
import quantumeGate as qg
import numpy as np


def oracle(state):
    toffoli = qg.ToffoliGate([0,1,2])
    phase = qg.PhaseGate([2], 3)
    inv_phase = qg.InversePhaseGate([2], 3)
    

    newstate = toffoli.apply(state)
    newstate = inv_phase.apply(newstate)
    newstate = toffoli.apply(newstate)
    newstate = inv_phase.apply(newstate)

    return newstate

def diffusion(state):
    n = 3
    for i in range(n-1):
        hadamard = qg.HadamardGate([i], n)
        state = hadamard.apply(state)
    for i in range(n-1):
        x = qg.PauliXGate([i], n)
        state = x.apply(state)
    
    hadamard = qg.HadamardGate([2], n)
    state = hadamard.apply(state)
    toffoli = qg.ToffoliGate([0,1,2])
    state = toffoli.apply(state)
    tdagger = qg.InverseTGate([2], n)
    state = tdagger.apply(state) 
    state = toffoli.apply(state)
    t = qg.TGate([2], n)
    state = t.apply(state)
    state = hadamard.apply(state)
    
    
    for i in range(n-1):
        x = qg.PauliXGate([i], n)
        state = x.apply(state)
    for i in range(n-1):
        hadamard = qg.HadamardGate([i], n)
        state = hadamard.apply(state)
    
    return state

s = ([0, 1, 0, 1, 0, 0, 0, 0])


iterations =  int((np.pi / 4 ) * 2**3)
print(iterations)   
for n in range(iterations):
    s = oracle(s)
    s = diffusion(s)
probabilities = np.abs(s)**2
print(probabilities)