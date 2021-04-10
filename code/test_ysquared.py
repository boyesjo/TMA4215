"""
    Trains the network on the function F(x) = x^2 / 2
    Then plots the exact function and network's for comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from network import Network

# initialises and trains network
test = Network(2, 5, 0.5)
y_train = 4 * (np.random.rand(5000) - 0.5)
c = y_train**2
test.train([y_train], c, 0.02, 1e-6, 5000)

# plots and saves exact solution vs network in the same plot
y0 = np.linspace(-2, 2, 1000)
F_ext = y0**2
F_net = test.forward([y0])
plt.plot(y0, F_ext, label="Exact")
plt.plot(y0, F_net, label="Network")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("$F(x)=x^2/2$")
plt.legend()
plt.savefig("1d.pdf")
