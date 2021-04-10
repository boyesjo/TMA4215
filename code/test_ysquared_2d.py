"""
    Trains the network on the function F(x, y) = (x^2 + y^2)/2
    Then plots the exact function and network's for comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from network import Network


# intialisationg of network and data, then trains
net = Network(4, 10, 0.2)
y_training = 4*(np.random.rand(2, 2000) - 0.5)  # random 2-tuples in (-2,2)^2
F_training = (np.sum(0.5*y_training**2, axis=0))
net.train(y_training, F_training, 0.01, 1e-4, 10_000)

# evaluates F and network on (-2,2)^2
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
F_ext = [[0.5*(xi**2 + yi**2) for yi in y] for xi in x]
F_net = [[net.forward(([xi], [yi]))[0] for yi in y] for xi in x]

fig, axes = plt.subplots(1, 2)

im = axes[0].imshow(F_ext, vmin=0, vmax=4, extent=(-2, 2, -2, 2))
axes[0].set_title("Exact solution")


im = axes[1].imshow(F_net, vmin=0, vmax=4, extent=(-2, 2, -2, 2))
axes[1].set_title("Network")

fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.021,  pad=0.04)
plt.savefig("2d.pdf")
