"""
    Implementation of the symplectic euler method and the størmer--verlet method for seperable hamiltonian systems
    Main function test both on a normalised nonlinear pendulum hamiltonian
"""

import numpy as np
import matplotlib.pyplot as plt


def sympectic_euler(q0, p0, dT, dV, T, h):
    d = np.size(q0)
    q_list = np.zeros((T, d)); q_list[0] = q0
    p_list = np.zeros((T, d)); p_list[0] = p0

    for i in range(T-1):
        q_list[i+1] = q_list[i] + h * dT(p_list[i])
        p_list[i+1] = p_list[i] - h * dV(q_list[i+1])

    return q_list, p_list


def stoermer_verlet(q0, p0, dT, dV, T, h):
    d = np.size(q0)
    q_list = np.zeros((T, d)); q_list[0] = q0
    p_list = np.zeros((T, d)); p_list[0] = p0

    for i in range(T-1):
        p_tmp = p_list[i] - 0.5 * h * dV(q_list[i])
        q_list[i+1] = q_list[i] + h * dT(p_tmp)
        p_list[i+1] = p_tmp - 0.5 * h * dV(q_list[i+1])

    return q_list, p_list


# test the methods on a nonlinear pendulum hamiltonian
# plots phase space and hamiltonian value along the trajectory
# units, mass etc are choses such such that the coeffiscients cancel
if __name__ == "__main__":

    # nonlinear pendulum hamiltonian
    dT = lambda p: p
    dV = lambda q: np.sin(q)
    H = lambda q, p: 0.5 * p**2 + (1 - np.cos(q))

    # inital condition and solver parameters
    q0 = 1
    p0 = 0
    h = 0.01
    T = 1000

    q_euler, p_euler = sympectic_euler(q0, p0, dT, dV, T, h)
    q_stoermer, p_stoermer = stoermer_verlet(q0, p0, dT, dV, T, h)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    plt.subplot(221)
    plt.plot(q_euler, p_euler)
    plt.scatter(q0, p0, label="Initial condition")
    plt.xlabel("$\\theta$")
    plt.ylabel("p")
    plt.title("Symplectic Euler method")

    plt.subplot(222)
    plt.plot(q_euler, p_euler)
    plt.scatter(q0, p0, label="Initial condition")
    plt.xlabel("$\\theta$")
    plt.title("Størmer-Verlet method")

    h_euler = plt.subplot(223)
    plt.plot(H(q_euler, p_euler))
    plt.ylabel("H")
    plt.xlabel("Time steps")

    plt.subplot(224, sharey=h_euler)
    plt.plot(H(q_stoermer, p_stoermer))
    plt.xlabel("Time steps")

    plt.savefig("ode.pdf")
