import numpy as np

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
        self.N = N
        self.size = 2 ** N # Determine the size of the state vector

        if initial_state is None:
            # Initialize to the |0...0> state if no initial state is provided
            self.state = np.zeros(self.size, dtype=complex)
            self.state[0] = 1.0
        elif isinstance(initial_state, int):
            # If initial_state is an integer, treat it as a basis state index
            self.state = np.zeros(self.size, dtype=complex)
            self.state[initial_state] = 1.0