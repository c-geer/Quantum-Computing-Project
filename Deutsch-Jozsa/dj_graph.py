from dj import deutsch_jozsa
from dj import func
import time
from matplotlib import pyplot as plt

if __name__ == "__main__":

    tc = []
    tb = []

    # Loop through a variety of n values
    for n in range(2, 10):  # Example range, adjust as needed
        type = ("constant", "balanced")
        for t in type:
            f = func(n, t)

            # Measure execution time
            t1 = time.time()
            is_constant = deutsch_jozsa(n, f)
            t2 = time.time()

            # append time_elapsed to array
            time_elapsed = t2 - t1
            if t == "constant":
                tc.append(time_elapsed)
            else:
                tb.append(time_elapsed)


            
