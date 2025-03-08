import numpy as np

class Tensor(object):
    """
    Class to represent a tensor
    """
    def __init__(self, data):
        """Constructor for the Tensor class

        Args:
            data (list): List representing the tensor, takes the form of a 2D list, i.e.
            [[1, 2], [3, 4], [5, 6]] for a 3x2 tensor
            [1,2] for a 1x2 tensor (column vector)
            
        """
        #Convert the data to a numpy array
        self.data = np.atleast_2d(np.array(data, dtype=np.complex128))
        self.shape = self.data.shape


    def __add__(self, other):
        """Add two tensors together

        Args:
            other (Tensor): Another tensor

        Returns:
            Tensor: Sum of the two tensors
        """
        return Tensor(self.data + other.data)
    
    def __repr__(self):
        """Return the string representation of the tensor

        Returns:
            str: String representation of the tensor
        """
        return f"{self.data}"
    
    def __getitem__(self, key):
        """
        Get the value of a specific element in the matrix
        
        Parameters:
        ------------
            key: Tuple (row, col) specifying the element
        """
        return self.data[key]
    
    def __setitem__(self, key, value):
        """
        Set the value of a specific element in the Tensor
        
        Parameters:
        ------------
            key: Tuple (row, col) specifying the element
            value: Value to set at the element
        """
        self.data[key] = value
    
    def __len__(self):
        """Compute the length of the tensor
        
        Returns:
            Integer: Length of data array
        """
        return len(self.data)
    
    def TensorProduct(self, other):
        """Compute the tensor product of two tensors

        Args:
            other (Tensor): Another tensor

        Returns:
            Tensor: Tensor product of the two tensors
        """
        shape = (self.shape[0] * other.shape[0], self.shape[1] * other.shape[1])
        data = np.zeros(shape, dtype=np.complex128)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data[i*other.shape[0]:(i+1)*other.shape[0], j*other.shape[1]:(j+1)*other.shape[1]] = self.data[i, j] * other.data
        data = data.astype(np.complex128)

        return Tensor(data)

#test xx
t1 = Tensor([[1, 1], [1, -1]])

t2 = Tensor([[1, 1], [1, -1]])
t3 = t1 + t2
t4 = t1.TensorProduct(t2)

#tensor product between 1
#                       2
#and
#5
#6
t5 = Tensor([[1, 2]])
t6 = Tensor([[5], [6]])

#print(t4)
'''
Tensor product between 
[1 2]   and [5 6]
[3 4]       [7 8]

should return:
[[ 5.  6. 10. 12.]
 [ 7.  8. 14. 16.]
 [15. 18. 20. 24.]
 [21. 24. 28. 32.]]

'''
