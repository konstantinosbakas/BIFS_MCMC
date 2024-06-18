import numpy as np
from Utils.fourier import Fourier
from scipy.stats import circmean

def polarToCartesian(rho, theta): 
    polar = rho * np.cos(theta), rho * np.sin(theta)
    return polar

def cartesianToPolar(a, b):
    cartesian = np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)
    return cartesian

def getPosteriorEstimates(chain, magnitude=True):
    if magnitude:
        est = np.mean(chain, axis=0)
    else:
        est = circmean(chain, axis=0)
    return est

def compute_residuals(true, result):
    return np.absolute(true - result)
