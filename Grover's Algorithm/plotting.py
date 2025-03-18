import numpy as np
import time
import matplotlib.pyplot as plt
from lazy_grover import grovers_algo
from bit_wise_lazy import grovers_algorithm_lazy
from bit_wise_grover import grovers_algorithm
from sparse_grover import grovers_algorithm_sparse

n_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14])
time_list_lazy2 = []
time_list_lazy1 = []
time_list_sparse = []
time_list = []
for n in n_list:
    marked_index = 2**n - n
    start_time = time.time()
    grovers_algorithm_lazy(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list_lazy2.append(total_time)
    """
    start_time = time.time()
    grovers_algorithm(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list.append(total_time)
    
    start_time = time.time()
    grovers_algorithm_sparse(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list_sparse.append(total_time)
    
    start_time = time.time()
    state = grovers_algo(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list_lazy1.append(total_time)"
    """

    print(f"Done with {n}")


   
time_list = np.array(time_list)
time_list_lazy2 = np.array(time_list_lazy2)
time_list_lazy1 = np.array(time_list_lazy1)
time_list_sparse = np.array(time_list_sparse)
#plt.plot(n_list, time_list, label= "Bit-Wise")
plt.plot(n_list, time_list_lazy2, label = "Bit Wise Lazy")
#plt.plot(n_list, time_list_lazy1, label = "Lazy")
#plt.plot(n_list, time_list_sparse, label = "Sparse")
plt.title("Time for the Fastest Implementation")
plt.xlabel("Number of Qubits")
plt.ylabel("Time Taken in Seconds")
plt.legend()
plt.show()