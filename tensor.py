import numpy as np

class Tensor(object):
    """ Hello this is my sexy tensor class where you can add and tensor product two tensors x"""
    def __init__(self, data):
        self.data = np.atleast_2d(np.array(data))

    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __repr__(self):
        return f"{self.data}"
    
    def TensorProduct(self, other):
        shape = (self.data.shape[0] * other.data.shape[0], self.data.shape[1] * other.data.shape[1])
        data = np.zeros(shape)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                data[i*other.data.shape[0]:(i+1)*other.data.shape[0], j*other.data.shape[1]:(j+1)*other.data.shape[1]] = self.data[i, j] * other.data

        return Tensor(data)

#test xx
t1 = Tensor([[1, 2], [3, 4]])
print(t1)
t2 = Tensor([[5, 6], [7, 8]])
t3 = t1 + t2
t4 = t1.TensorProduct(t2)
print(t4)
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
