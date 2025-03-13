import numpy as np
import time
import matplotlib.pyplot as plt
from lazygrover1 import grovers_algo
from lazygrover2 import grovers_algorithm_lazy
from govers import grovers_algorithm

n_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14])
time_list_lazy2 = []
time_list_lazy1 = []
time_list = []
for n in n_list:
    marked_index = 2**n - n
    start_time = time.time()
    state = grovers_algorithm_lazy(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list_lazy2.append(total_time)

    start_time = time.time()
    state = grovers_algorithm(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list.append(total_time)

n_list = np.array([1,2,3,4,5,6,7])

for n in n_list:
    marked_index = 2**n - n
    start_time = time.time()
    state = grovers_algo(n, marked_index)
    end_time = time.time()
    total_time = end_time-start_time
    time_list_lazy1.append(total_time)

    
time_list = np.array(time_list)
time_list_lazy2 = np.array(time_list_lazy2)
time_list_lazy1 = np.array(time_list_lazy1)
plt.plot(n_list, time_list, label= "Bit-Wise Grover's Implementation")
plt.plot(n_list, time_list_lazy2, label = "Bit Wise Lazy Grover's Implementation")
plt.plot(n_list, time_list_lazy1, label = "Lazy Grover's Implementation")
plt.xlabel("Number of Qubits")
plt.ylabel("Time Taken in Seconds")
plt.legend
plt.show()