import numpy as np

class Lazy_Matrix(object):
    """
    General lazy matrix superclass
    -------
    Methods
    -------
    lazy_gate: obtains each element of some lazy matrix subclass gate
    apply: applies the lazy gate to a quantum register
    element: yields elements of the identity matrix
    """

    def __init__(self, n):
        """
        constructor of the lazy matrix class
        -------
        Inputs
        -------
        n: integer, dimension of desired matrix. 
        ->MUST be determined from the quantum register that the gate is being applied to!!!
        ->Note: n is the size of the quantum register, 2**N
        """
        assert n > 0 and bin(n).count('1') == 1 # check n corresponds to the size of the quantum register
        self.n = n

    
    def element(i, j, n):
        """
        lazily obtain individual elements of the identity matrix. 
        I am choosing identity because this will be overriden by subclasses anyway.
        Please do not use lazy matrix to apply the identity because that would be a colossal waste of time.
        -------
        Inputs
        -------
        i: integer, row
        j: integer, column
        n: integer, dimension of matrix
        -------
        Returns
        -------
        integer, corresponding element of identity matrix
        """
        if i == j:
            return 1
        else:
            return 0


    def lazy_gate(self):
        """
        method to yield every element of the lazy matrix, one at a time 
        (I think this is how they work please don't be mad if this is stupid)
        -------
        yields each element of the lazy matrix
        """
        for i in range(self.n):
            for j in range(self.n):
                yield self.element(i, j, self.n)

    
    def apply(self, Q)
        """
        method to apply the lazy matrix to our quantum register
        -------
        Inputs
        -------
        Q: object, the quantum register we want to apply the gate to
        -------
        Actions
        -------
        transforms the state of the quantum register through the gate
        """ 
        lazy_G = self.lazy_gate() # each element of the lazy matrix
        new_state = np.zeros(self.n, dtype=complex) # variable to hold the state of the transformed quantum register

        for i, value in enumerate(lazy_G):
            """
            Guys if you see this and think of a more efficient way of doing it please let me know. 
            I'm currently wondering whether there would be an advantage to evaluating the transformation
            by looping through rows and taking scalar products instead of individual elements and multiplying,
            but I'm not sure if it's worth it with lazy_G being a generator object and all.
            """
            row, col = divmod(i, n)
            new_state[row] += value * Q.state[col]

        Q.state = new_state
        Q.normalise() # ensure quantum register is normalised regardless of whether we've bothered with prefactors when constructing our gates