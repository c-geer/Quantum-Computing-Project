import numpy as np

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
        self.shape = (rows, cols)

    def from_dense(self, matrix):
        """
        Initialize a sparse matrix from a dense matrix
        
        Parameters:
        ------------
            matrix: 2D list or numpy array representing the dense matrix
        """
        if isinstance(matrix, np.ndarray):
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            self.rows = len(matrix)
            self.cols = len(matrix[0])
        
        self.shape = (self.rows, self.cols)
        self.data = {}  # Reset data
        
        for i in range(self.rows):
            for j in range(self.cols):
                if matrix[i][j] != 0:
                    self.data[(i, j)] = matrix[i][j]
        
        return self

    @classmethod
    def from_dense_matrix(cls, matrix):
        """
        Create a SparseMatrix from a dense numpy array or 2D list.
        
        Parameters:
        -----------
        matrix : numpy.ndarray or 2D list
            The dense matrix to convert
            
        Returns:
        --------
        SparseMatrix
            A sparse representation of the input matrix
        """
        if isinstance(matrix, np.ndarray):
            rows, cols = matrix.shape
        else:
            rows = len(matrix)
            cols = len(matrix[0])
        
        sparse_matrix = cls(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] != 0:
                    sparse_matrix.data[(i, j)] = matrix[i][j]
        
        return sparse_matrix

    def to_dense(self):
        """
        Convert the sparse matrix to a dense numpy array.
        
        Returns:
        --------
        numpy.ndarray
            Dense representation of the sparse matrix
        """
        matrix = np.zeros((self.rows, self.cols))
        for (i, j), value in self.data.items():
            matrix[i, j] = value
        return matrix

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
        """
        Multiply this sparse matrix with another matrix, vector, or scalar
            
        Parameters:
        ------------
            other: Matrix, vector, or scalar to multiply with

        Returns:
        ------------
            SparseMatrix or numpy.ndarray: Result of the multiplication
        """
        # For multiplying the matrix with a scalar
        if isinstance(other, (int, float, complex)): 
            result = SparseMatrix(self.rows, self.cols)
            for (row, col), value in self.data.items():
                result[row, col] = value * other
            return result
        
        # For multiplying with a 1D numpy array (matrix-vector product)
        elif isinstance(other, np.ndarray) and other.ndim == 1:
            if self.cols != len(other):
                raise ValueError(f"Cannot multiply matrix of shape ({self.rows}, {self.cols}) with vector of shape ({len(other)},)")
            
            result = np.zeros(self.rows)
            
            for (i, j), value in self.data.items():
                result[i] += value * other[j]
                
            return result
            
        # For multiplying with another sparse matrix
        elif isinstance(other, SparseMatrix):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions don't match for multiplication")
            
            # Multiply the matrices
            result = SparseMatrix(self.rows, other.cols)
            # Only iterates through the nonzero elements
            for (i, k), v1 in self.data.items():
                for (k2, j), v2 in other.data.items():
                    if k == k2: # Checks if the middle element is the same
                        result[i, j] = result[i, j] + v1 * v2
            return result
            
        else:
            raise TypeError("Multiplication is only supported with scalars, 1D numpy arrays, or another SparseMatrix")

    def __rmul__(self, other):
        """
        Right multiplication by a scalar.
        
        Parameters:
        -----------
        other : float
            Scalar to multiply with
            
        Returns:
        --------
        SparseMatrix
            Result of multiplication
        """
        if isinstance(other, (int, float, complex)):
            result = SparseMatrix(self.rows, self.cols)
            for (row, col), value in self.data.items():
                result[row, col] = value * other
            return result
        else:
            raise TypeError("Right multiplication is only supported with scalars")

    def tensor_product(self, other):
        """
        Compute the tensor (Kronecker) product of two sparse matrices.
        
        Parameters:
        -----------
        other : SparseMatrix
            The matrix to compute tensor product with
            
        Returns:
        --------
        SparseMatrix
            Tensor product of the two matrices
        """
        if not isinstance(other, SparseMatrix):
            raise TypeError("Can only compute tensor product with another SparseMatrix")
        
        m1, n1 = self.rows, self.cols
        m2, n2 = other.rows, other.cols
        
        result = SparseMatrix(m1 * m2, n1 * n2)
        
        for (i1, j1), v1 in self.data.items():
            for (i2, j2), v2 in other.data.items():
                result[(i1 * m2 + i2, j1 * n2 + j2)] = v1 * v2
        
        return result

    def __repr__(self):
        """String representation of the sparse matrix"""
        nnz = len(self.data)  # Number of non-zero elements
        return f"SparseMatrix(shape=({self.rows}, {self.cols}), nnz={nnz})"
    
    def display(self):
        """Display the sparse matrix in a readable format"""
        if len(self.data) == 0:
            return "Empty sparse matrix"
            
        output = [f"SparseMatrix shape=({self.rows}, {self.cols}), non-zero elements:"]
        sorted_items = sorted(self.data.items(), key=lambda x: (x[0][0], x[0][1]))
        
        for (i, j), value in sorted_items:
            output.append(f"  ({i}, {j}): {value}")
            
        return "\n".join(output)
    

if __name__ == "__main__":

        # Create a sparse matrix
        A = SparseMatrix(3, 3)
        A[0, 0] = 1
        A[1, 1] = 2
        A[2, 2] = 3

        # Create from a dense matrix
        dense = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
        B = SparseMatrix.from_dense_matrix(dense)
        print(B.display())

        # Multiply by a vector
        vec = np.array([1, 2, 3])
        result_vec = A * vec  # Returns: array([1, 4, 9])
        print(result_vec)

        # Matrix multiplication
        C = A * B  # Returns a new SparseMatrix

        # Tensor product
        D = A.tensor_product(B)
        print(D.display())
        print(D.data.items()[])
