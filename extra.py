import numpy as np
from scipy.stats import f
"""
Example code:
Defines the distribution of duration of a process
"""
def app_time(x, dfn, dfd, a, b):
    mean = 0.0
    dist = np.divide(f.pdf(x, dfn, dfd), (f.cdf(b, dfn, dfd) - f.cdf(a, dfn, dfd))) # f-dist for duration, truncated from a to b
    dist = np.divide(dist, np.sum(dist)) # normalization

    for item in zip(x, dist): mean = mean + (item[0] * item[1]) # expectation of duration

    return dist, mean

"""
Example code:
Defines the distribution of consumption rate
"""
def app_consumption(x, dfn, dfd, a, b):
    mean = 0.0
    dist = np.divide(f.pdf(x, dfn, dfd, scale=0.1), (f.cdf(b, dfn, dfd, scale=0.1) - f.cdf(a, dfn, dfd, scale=0.1))) # f-dist for duration, truncated from a to b
    dist = np.divide(dist, np.sum(dist)) # normalization

    for item in zip(x, dist): mean = mean + (item[0] * item[1]) # expectation of consumption

    return dist, mean

"""
Example code:
Reads a standard load profile and resize it to the desired length
"""
def read_slp(t, file):
    original_signal = np.genfromtxt(file, delimiter=',')

    slp_year = original_signal[:, 3]
    slp_avg_day = np.zeros(96)
    for i in range(1, 365):
        slp_avg_day = slp_avg_day + slp_year[((i-1)*96):((i)*96)]

    slp_avg_day = slp_avg_day / 365.0

    return slp_avg_day
