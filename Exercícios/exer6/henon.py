import numpy as np

def henon(a, y0, x0, n):
    x = np.zeros(n)
    x[0] = x0
    for n in range(0, n-1, 1):
        x[n+1] = y0 + 1.0 - a *x[n]*x[n]
    return x
