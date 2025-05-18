# code to compute RMSE and spectral estimate for the given data

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd
from scipy.linalg import norm
from scipy.linalg import pinv
from scipy.linalg import svdvals
from scipy.linalg import eig


def compute_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def compute_spectral_estimate(data):
    u, s, vh = svd(data)
    return s

def compute_svd(data):
    u, s, vh = svd(data)
    return u, s, vh

def compute_svd_norm(data):
    u, s, vh = svd(data)
    return norm(s)

def compute_svd_pinv(data):
    u, s, vh = svd(data)
    return pinv(data)

def compute_svdvals(data):
    u, s, vh = svd(data)
    return svdvals(data)

def compute_eigenvalues(data):
    w, v = eig(data)
    return w

def compute_eigenvectors(data):
    w, v = eig(data)
    return v

def compute_eigenvalues_and_vectors(data):
    w, v = eig(data)
    return w, v


