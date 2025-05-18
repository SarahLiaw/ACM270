import numpy as np
import matplotlib.pyplot as plt


# inspired by ACM 106a Homework 6 code
sigma = 10
rho = 28
beta = 8/3

def lorenz(t, X):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

T = 50
h = 0.00625

u_prev = 0
t_prev = 0

t = np.arange(0, T + h, h)
X = np.zeros((3, len(t)))
X[:, 0] = [0, 1, 1]

for n in range(len(t) - 1):
    k0 = lorenz(t[n], X[:, n])
    k1 = lorenz(t[n] + h/2, X[:, n] + h * k0 / 2)
    k2 = lorenz(t[n] + h/2, X[:, n] + h * k1 / 2)
    k3 = lorenz(t[n] + h,   X[:, n] + h * k2)
    X[:, n+1] = X[:, n] + (h/6) * (k0 + 2*k1 + 2*k2 + k3)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(X[0, :], X[1, :], X[2, :], lw=0.5)
ax.set_title(f"Lorenz Attractor (h = {h})")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True)
plt.show()