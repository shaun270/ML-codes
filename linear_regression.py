from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 10
a = 0.0001
W = np.zeros(y-est)

def cost_function(y_est, y_real):
    return y_est - y_real

def prediction(X, W, b):
    return X@W + b

def gradient_descent(n, X, W, b, y_real, a):
    for i in range(n):
        y_est = prediction(X, W, b)
        diff = cost_function(prediction(X, W, b),  y_real)
        dw = -2/n * np.sum(X@diff)
        db = -2/n * np.sum(diff)

        w = w - a * dw
        b = b - a * db
    
    return w_fin, b_fin

y_pred = 
    



