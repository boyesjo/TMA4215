"""
    Visualises some of the given data
    P(q) appears exact, while T(p) may be noisy
"""

import numpy as np
import matplotlib.pyplot as plt
from project_2_data_acquisition import concatenate

data = concatenate()
q = data["Q"]
p = data["P"]
V = data["V"]
T = data["T"]

plt.subplots_adjust(wspace=0.4)

plt.subplot(121)
plt.scatter(np.sum(q**2, axis=0), V, s=0.6, marker=",")
plt.xlabel("$q^2$")
plt.ylabel("V")
plt.title("Potential as a function of $q^2$")

plt.subplot(122)
plt.scatter(np.sum(p**2, axis=0), T, s=0.6, marker=",")
plt.xlabel("$p^2$")
plt.ylabel("T")
plt.title("Kinetic energy as a function of $p^2$")
plt.savefig("givendata.png")