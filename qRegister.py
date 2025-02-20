import numpy as np
import random

class Q_Register(object):
    """
    The quantum register class stores a collection of qbits as a big vector
    """
    def __init__(self, N, initial_state):
        """
        Constructor
        -------
        Inputs
        -------
        N: Integer, Number of qubits
        initial_state: Integer, the initial state of the system, like |0> or |7> or |8123>
        """
        self.N = N # assertion unnecessary, size is computed automatically from #qubits
        self.size = 2 ** N # Determine the size of the state vector

        if initial_state is None:
            # Initialize to the |0...0> state if no initial state is provided
            self.state = np.zeros(self.size, dtype=complex)
            self.state[0] = 1.0

        elif isinstance(initial_state, int):
            # If initial_state is an integer, treat it as a basis state index
            self.state = np.zeros(self.size, dtype=complex)
            self.state[initial_state] = 1.0


    def normalise(self):
        """
        Method to normalise the state of the quantum register
        -------
        state --> normalised state
        """
        mag = np.linalg.norm(self.state)
        self.state /= mag


    def initial_H(self):
        """
        Method to perform the initial Hadamard transform at the beginning of Grover's algorithm 
        to put the quantum register into the equal superposition of all N states
        -------
        state --> equal superposition of all states
        """
        self.state = np.ones(self.size)
        self.normalise()


    def __str__(self):
        """
        Define print of Q_Register class to display current state
        """
        return str(self.state)


    def measure(self):
        """
        measure quantum register state
        -------
        Returns
        -------
        state from 0 to N-1: integer
        """
        probs = np.abs(self.state)**2
        index = random.choices(range(len(probs)), probs)[0]

        return index
    


#test_Q = Q_Register(3, 6)
#print(test_Q.measure())