"""
    Solves the Hamiltonian dynamics with some well trained network (from train_hamiltonian.py)
    Plots q^2 over time along with Hamiltonian over time
"""

import numpy as np
import matplotlib.pyplot as plt
from network import Network
import pickle
from ode import stoermer_verlet
from project_2_data_acquisition import generate_data


with open("trained_networks", "rb") as save_file:

    V, T = pickle.load(save_file)

    data = generate_data(30) 
    q_ext = data["Q"]
    q0 = (data["Q"].T)[0]
    p0 = (data["P"].T)[0]

    steps = 1000  # doesnt need to be this high
    t = np.linspace(0, data["t"][-1], steps)
    h = t[1] - t[0]
    H0 = data["V"][0] + data["T"][0]  # hamiltonian at t=0

    q_net, p_net = stoermer_verlet(q0, p0, T.grad, V.grad, steps, h) 

    plt.subplot(211)
    plt.plot(t, np.sum(q_net**2, axis=1), label="Network")
    plt.ylabel("$q^2$")
    plt.plot(data["t"], np.sum(q_ext**2, axis=0), label="Given data")
    plt.title("Time propegation of position")
    plt.xticks([],[])

    plt.subplot(212)
    plt.plot(t, V.forward(q_net.T) + T.forward(p_net.T), label="Network")
    plt.plot(data["t"], data["V"] + data["T"], label="Given data")
    plt.ylim(0, 2*H0) 
    plt.yticks([0, H0, 2*H0], ["0","1","2"])
    plt.ylabel("$H(t)/H_0$")
    plt.title("Time propegation of Hamiltonian")
    plt.xlabel("$t$")
    plt.legend()
    plt.savefig("ham_ode.pdf")
