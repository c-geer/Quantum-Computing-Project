import time
from dj import func

def classical_dj(n, f):
    isConstant = True
    hasZero = False
    hasOne = False

    for i in range(2**n // 2 + 1):

        if f[i] == 0:
            hasZero = True
        else:
            hasOne = True

        if hasZero and hasOne:
            isConstant = False
            return isConstant
    
    return isConstant
    

if __name__ == "__main__":

    # Open a file to store the results
    with open("Deutsch-Jozsa/classical_results.txt", "w") as file:
        # Loop through a variety of n values
        for n in range(31, 33):  # choose arbitrary limit
            type = ("constant", "balanced")
            for t in type:
                f = func(n, t)

                # Measure execution time
                t1 = time.time()
                is_constant = classical_dj(n, f)
                t2 = time.time()

                # Determine time taken and write results to file
                time_elapsed = t2 - t1
                file.write(f"n: {n}, type: {t}, time: {time_elapsed:.6f} seconds\n")