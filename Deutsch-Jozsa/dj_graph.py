from dj import deutsch_jozsa
from dj import func
import time
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Open a file to store the results
    with open("time_results.txt", "w") as file:
        # Loop through a variety of n values
        for n in range(1, 10):  # takes too long for n>10
            type = ("constant", "balanced")
            for t in type:
                f = func(n, t)

                # Measure execution time
                t1 = time.time()
                is_constant = deutsch_jozsa(n, f)
                t2 = time.time()

                # Determine time taken and write results to file
                time_elapsed = t2 - t1
                file.write(f"n: {n}, type: {t}, time: {time_elapsed:.6f} seconds\n")



            
