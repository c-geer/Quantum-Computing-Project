import numpy as np
from lazyMatrix import Lazy_Matrix

class Lazy_Hadamard(Lazy_Matrix):
    """
    subclass of Lazy_Matrix, lazy definition of the Hadamard gate
    -------
    Methods
    -------
    element: overrides element definition of superclass, returns Hadamard matrix elements
    apply: bitwise application of the lazy Hadamard to some quantum state
    """

    def element(i, j, n):
        """
        lazily obtain individual elements of the Hadamard matrix
        -------
        Inputs
        -------
        i: integer, row
        j: integer, column
        n: integer, dimension of matrix
        -------
        Returns
        -------
        integer, corresponding element of Hadamard matrix
        """
        if n == 1:
            return 1
        else:
            i_half = i // 2
            j_half = j // 2
            sign = 1 if (i % 2 == 0) or (j % 2 == 0) else -1
            return sign * element(i_half, j_half, n // 2)

    
    def apply(self, Q):
        """
        Apply Hadamard gate lazily and normalize
        -------
        Inputs
        -------
        Q: object, The quantum register
        -------
        Actions
        -------
        Q.state --> Hadamard transformed state
        ----------
        1 << bit: binary representation of 1 is shifted left by bit positions
        eg.
        1 << 2 = 100, because, 001 << 2 = 100 (1 is 001, 1 shifted left by 2 positions)
        3 << 2 = 1100, because, 011 << 2 = 1100 (3 is 011, 3 shifted left by 2 positions)

        a ^ b: a XOR b
        eg.
        3 ^ 1 = 2, because, 011 ^ 001 = 010 (where the bits are different, the result is 1, otherwise 0)

        Putting it together, 
        i ^ (1 << bit) flips the bit of i at the position specified by bit
        Note that we count from right starting from 0
        bit = 011101
        pos = 543210

        eg.
        5 = 101

        5 ^ (1 << 1) = 111 = 7
        5 ^ (1 << 0) = 101 = 5
        5 ^ (1 << 2) = 001 = 1

        Proof:
        5 ^ (1 << 1) = 101 ^ (001 << 1) = 101 ^ 010 = 111
        5 ^ (1 << 0) = 101 ^ (001 << 0) = 101 ^ 001 = 100
        5 ^ (1 << 2) = 101 ^ (001 << 2) = 101 ^ 100 = 001
        """
        new_state = np.zeros_like(Q.state, dtype=np.complex128)

        for i in range(Q.size):
            for bit in range(Q.N):
                new_state[i] += state[i ^ (1 << bit)]  # Apply bitwise Hadamard

        Q.state = new_state
        Q.normalise()