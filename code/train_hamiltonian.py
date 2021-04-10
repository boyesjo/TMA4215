"""
    Trains network to model the unknown hamiltonians
    Uses stochastic gradient descent with the adams method
    Saves to disk to allow ODE-solving without retraining
"""

import numpy as np
from project_2_data_acquisition import generate_data, concatenate
import matplotlib.pyplot as plt
from network import Network
from ode import stoermer_verlet

# import data
data_train = concatenate()
q_train = data_train["Q"]
p_train = data_train["P"]
V_train = data_train["V"]
T_train = data_train["T"]

V = Network(6, 32, 0.05)
T = Network(6, 32, 0.05)

plt.plot(V.stoch_train(q_train, V_train, batch_size=4096, learning_param=0.002, min_error=1e-5, max_iter=20_000))
plt.title("Convergence of $V$ when training")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.yscale("log")
plt.savefig("V_convergence.pdf")
plt.clf()

plt.plot(T.stoch_train(p_train, T_train, batch_size=4096, learning_param=0.01, min_error=1e-5, max_iter=10_000))
plt.yscale("log")
plt.title("Convergence of $T$ when training")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.savefig("T_convergence.pdf")


# saves to disk
import pickle
with open("trained_networks", "wb") as save_file:
    pickle.dump([V, T], save_file)
