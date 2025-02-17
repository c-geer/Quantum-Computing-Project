import numpy as np
from lazyMatrix import Lazy_Matrix

class Lazy_Hadamard(Lazy_Matrix):
    """
    subclass of Lazy_Matrix, lazy definition of the Hadamard gate
    -------
    Methods
    -------
    element: overrides element definition of superclass, returns Hadamard matrix elements
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