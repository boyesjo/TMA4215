"""
    Trains the network on a nonlinear pendulum hamiltonian
    Then uses the stoermer-verlet with both known gradients and netowrk gradients
     to time propegate for some initial condition
    Plots time propegation, hamiltonian, and angle and momentum and their derivatives
"""

import numpy as np
import matplotlib.pyplot as plt
from ode import stoermer_verlet
from network import Network

# intervals to train on
q_min = -2; q_max = 2; q = np.linspace(q_min, q_max); q_data = 4 * (np.random.rand(2000) - 0.5)
p_min = -2; p_max = 2; p = np.linspace(p_min, p_max); p_data = 4 * (np.random.rand(2000) - 0.5)

V_ext = 1 - np.cos(q_data)
T_ext = 0.5 * p_data**2

# network initialisation and training
V = Network(2, 6, 0.4)
T = Network(2, 6, 0.4)
V.train([q_data], V_ext, learning_param=0.005, min_error=1e-5, max_iter=10000)
T.train([p_data], T_ext, learning_param=0.002, min_error=1e-5, max_iter=20000) # T seems to converge slower

# intial conditions and other parameters for the ode
q0 = np.array([1])
p0 = np.array([1])
H0 = (0.5*p0**2 + 1 - np.cos(q0))[0]
steps = 3000
h = 0.01

# solves ODE using trained energy function netowrks
q_net, p_net = stoermer_verlet(q0, p0, T.grad, V.grad, steps, h)
# using exact hamilotnian
q_ext, _ = stoermer_verlet(q0, p0, lambda p: p, lambda q: np.sin(q), steps, h)


plt.subplots_adjust(wspace=0.3, hspace=0.6)

plt.subplot(311)
plt.title("Angle over time")
plt.plot(q_net, label="Network")
plt.plot(q_ext, label="Exact")
plt.legend(loc=(0,-3))

plt.subplot(334)
plt.plot(V.forward(q_net.T) + T.forward(p_net.T))
plt.yticks([0, H0, 2*H0], ["0", "1", "2"])
plt.ylabel("H(t)/H(0)")
plt.title("H(t)")

plt.subplot(335)
plt.plot(q, V.forward([q]))
plt.plot(q, 1 - np.cos(q))
plt.title("$V(\\theta)$")

plt.subplot(336)
plt.plot(p, T.forward([p]))
plt.plot(p, 0.5 * (p)**2)
plt.title("$T(p)$")

plt.subplot(338)
plt.plot(q, [V.grad([x]) for x in q])
plt.plot(q, np.sin(q))
plt.title("$\\partial V / \\partial \\theta$")

plt.subplot(339)
plt.plot(p, [T.grad([x]) for x in p])
plt.plot(p, p)
plt.title("$\\partial T / \\partial p$")

plt.savefig("pendulum.pdf")
