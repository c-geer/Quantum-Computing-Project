class SparseMatrix:
    def __init__(self, rows, cols):
        """
        Initialize a sparse matrix with the given number of rows and columns
        
        Parameters:
        ------------
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.data = {}  # Dictionary to store nonzero values

    def __setitem__(self, key, value):
        """
        Set the value of a specific element in the matrix
        
        Parameters:
        ------------
            key: Tuple (row, col) specifying the element
            value: Value to set at the element
        """
        row, col = key
        if value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]
    
    def __getitem__(self, key):
        """
        Get the value of a specific element in the matrix
        
        Parameters:
        ------------
            key: Tuple (row, col) specifying the element
        """
        row, col = key
        return self.data.get((row, col), 0)

    def __mul__(self, other):
        """"
        Multiply this sparse matrix with another matrix or scalar
            
        Parameters:
        ------------
            other: Matrix or scalar to multiply with

        Returns:
        ------------
            SparseMatrix: Result of the multiplication
        """
        #For multiplying the matrix with a scalar
        if isinstance(other, (int, float, complex)): #Checks if the matrix is actually a scalar 
            result = SparseMatrix(self.rows, self.cols)
            for (row, col), value in self.data.items():
                result[row, col] = value * other
            return result
        
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        #Multiply the matrices
        result = SparseMatrix(self.rows, other.cols)
        #Only iterates through the nonzero elements
        for (i, k), v1 in self.data.items():
            for (k2, j), v2 in other.data.items():
                if k == k2: #Checks if the middle element is the same
                    result[i, j] = result[i, j] + v1 * v2
        return result
