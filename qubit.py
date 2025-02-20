import numpy as np
import random

class Qubit(object):
    """
    represents a single qubit, can be in any combination of the 0 and 1 state.
    -------
    Methods
    -------
    normalise: normalise the qubit state
    measure: measure the qubit state
    gate: apply a gate to the qubit state
    """
    
    
    def __init__(self, alpha, beta):
        """
        qubit contructor
        -------
        Inputs
        -------
        alpha: amplitude of the 0 state
        beta: amplitude of the 1 state
        -------
        zero: 0 state
        one: 1 state
        state: qubit state in vector form
        """
        self.zero = np.array([1, 0], dtype=complex)
        self.one = np.array([0, 1], dtype = complex)
        
        self.state = alpha * self.zero + beta * self.one
        self.normalise() # normalise the input state
        
        
    def normalise(self):
        """
        method to normalise qubit state
        """
        
        norm = np.linalg.norm(self.state)
        self.state /= norm
        
        
    def measure(self):
        """
        measure qubit state
        -------
        Returns
        -------
        0 or 1: integer
        """
        probs = self.state**2
        index = random.choices(range(len(probs)), probs)[0]

        return index
        
        
    def apply_gate(self, gate):
        """
        apply a gate to a single qubit
        -------
        Inputs
        -------
        gate: gate to be applied
        """
	    # gate is an object of the gate class, eg. Hadamard gate
        self.state = gate.apply(self.state, 1)
