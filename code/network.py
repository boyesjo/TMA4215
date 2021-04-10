"""
    General neural network class
    Activation and hypothesis functions are replacable
"""


import numpy as np


def sigma(x):
    return np.tanh(x)


def sigma_prime(x):
    return 1/(np.cosh(x)**2)


def eta(x):
    return x
    #return 0.5 * (np.tanh(0.5*x) + 1)


def eta_prime(x):
    return np.ones_like(x)
    #return 0.5/(np.cosh(x) + 1)


class Network():

    def __init__(self, dimension, layers, step_size):
        # network properties
        self.d = dimension
        self.K = layers
        self.h = step_size

        # initial random weights and biases
        self.W = np.random.randn(layers, dimension, dimension)
        self.b = np.random.randn(layers, dimension, 1)
        self.w = np.random.randn(dimension)
        self.mu = np.random.randn()

        # input dependent variables
        self.Z = None
        self.Y = None
        self.I = None
        self.dtheta = None

    # forward propegation given some y0 \in \R^{d_0 x I} where d0 <= d
    def forward(self, y0):
        self.I = np.size(y0[0])
        self.Z = np.zeros((self.K + 1, self.d, self.I))

        # pads input with zeros
        self.Z[0] = np.pad(y0, ((0, self.d - len(y0)), (0,0)), "constant")

        # calculates all Z_k
        for i in range(self.K):
            self.Z[i+1] = self.Z[i] + self.h * sigma(self.W[i] @ self.Z[i] + self.b[i])

        self.Y = eta(self.w @ self.Z[-1] + self.mu)
        return self.Y


    # forward and backward progates
    # intermediate Z-values are saved as member variables
    # returns error
    def sweep(self, y0, c):
        self.forward(y0)

        P = np.zeros_like(self.Z)
        P[-1] = np.outer(self.w, (self.Y - c) * eta_prime(self.w @ self.Z[-1] + self.mu))

        for i in range(self.K - 1, 0, -1):
            tmp = self.W[i].T @ (eta_prime(self.W[i] @ self.Z[i] + self.b[i]) * P[i+1])
            P[i] = P[i+1] + self.h * tmp

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dw = self.Z[-1] @ ((self.Y - c) * eta_prime(self.w @ self.Z[-1] + self.mu))
        dmu = eta_prime(self.w @ self.Z[-1] + self.mu) @ (self.Y - c)

        for i in range(self.K):
            tmp = P[i + 1] * sigma_prime(self.W[i] @ self.Z[i] + self.b[i])
            dW[i] = self.h * tmp @ self.Z[i].T
            db[i] = np.sum(self.h * tmp)

        self.dtheta = np.array([dW, db, dw, dmu], dtype="object")

        # arbitrary meassure of error, but consistent over the same data
        return np.average((self.Y - c)**2) 


    # trains on some data set (y0, c)
    # using the adams method
    # returns list of error after each iteration
    def train(self, y0, c, learning_param=0.01, min_error=0.01, max_iter=2000):
        error = [] # list of errors after each iteration

        # adams method constats
        b1 = 0.9; b2 = 0.999; e = 1e-8
        a = learning_param

        # previous iteration variables
        v_old = 0
        m_old = 0

        for i in range(1, int(max_iter) + 1):
            error.append(self.sweep(y0, c))
            print(f"Iteration: {i}, error: {error[-1]}") 

            if error[-1] < min_error:
                break

            m = b1 * m_old + (1 - b1) * self.dtheta
            v = b2 * v_old + (1 - b2) * (self.dtheta * self.dtheta)
            m_hat = m/(1 - b1**i)
            v_hat = v/(1 - b2**i)
            v_hat_sqrt = np.array([np.sqrt(el) for el in v_hat], dtype="object") # brute force way of component-wise sqrt 

            # updates whole set of weights, biases etc.
            self.W, self.b, self.w, self.mu = np.array([self.W, self.b, self.w, self.mu], dtype="object") - a * m_hat / (v_hat_sqrt + e)

            m_old = m
            v_old = v
        
        return error

    
    def stoch_train(self, y0, c0, batch_size, learning_param=0.01, min_error=0.01, max_iter=1000):
        error = []

        b1 = 0.9; b2 = 0.999; e = 1e-8; a = learning_param
        v_old = 0; m_old = 0

        y = y0  # to hold randomly ordered y0
        c = c0
        n = len(c0)

        for i in range(1, max_iter+1): 

            # shuffles y0 and c in the same way
            if (n+1)*batch_size > len(c0):
                random_order = np.random.permutation(len(c0))
                y = np.array(y0).T[random_order]
                c = c0[random_order]
                n = 0

            # draws batch_size elements from shuffled c and y 
            batch_y = np.transpose(y[n*batch_size : ((n+1)*batch_size)])
            batch_c = c[n*batch_size : (n+1)*batch_size]


            # the rest is the same as general adams descent
            error.append(self.sweep(batch_y, batch_c))
            print(f"Iteration: {i}, error: {error[-1]}") 

            if error[-1] < min_error:
                break

            m = b1 * m_old + (1 - b1) * self.dtheta
            v = b2 * v_old + (1 - b2) * (self.dtheta * self.dtheta)
            m_hat = m/(1 - b1**i)
            v_hat = v/(1 - b2**i)
            v_hat_sqrt = np.array([np.sqrt(el) for el in v_hat], dtype="object") 

            self.W, self.b, self.w, self.mu = np.array([self.W, self.b, self.w, self.mu], dtype="object") - a * m_hat / (v_hat_sqrt + e)

            m_old = m
            v_old = v
            n += 1

        return error


    # returns mathematically exact gradient of network for some input y
    def grad(self, y0):
        y = np.array(y0)
        self.forward(y.reshape(-1,1))  # first caculates Zk-s
        A = eta_prime(self.w @ self.Z[-1] + self.mu) * self.w

        for i in range(self.K-1, -1, -1):
            A = A + self.W[i].T @ (self.h * sigma_prime(self.W[i] @ self.Z[i] + self.b[i]).reshape(self.d) * A)

        return A[0:np.size(y[0])]  # "unpads" the vector back down to d0
