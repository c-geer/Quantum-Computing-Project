from matplotlib import pyplot as plt
import numpy as np

# Open the file and read its contents
with open("Deutsch-Jozsa/classical_results.txt", "r") as file:
    lines = file.readlines()

# Initialize empty lists for times
times = []

# Process each line
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace
    if line:  # Ignore empty lines
        try:
            # Extract time
            time_part = line.split(",")[-1]
            time = float(time_part.split(":")[1].split()[0].strip())
            times.append(time)
        except (IndexError, ValueError):
            print(f"Skipping malformed line: {line}")

n = np.arange(1, 31, 1) # list of number of qubits values
tc = times[0::2]
tb = times[1::2]

plt.plot(n, tc, color="green", label="f(x) = constant")
plt.plot(n, tb, color="blueviolet", label="f(x) = balanced")
plt.title("Classical Algorithm: Time elapsed vs. Number of qubits")
plt.xlabel("Number of qubits")
plt.ylabel("Time taken to determine function type [s]")
plt.legend()
plt.show()




