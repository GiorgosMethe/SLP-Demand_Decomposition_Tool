import lpd
import extra
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

"""
Main function of test python module
"""
def main():
    random.seed(os.urandom(967)) # initialize random generator
    t = np.linspace(0.0, 24.0, 96.0) # define the time axis of a day, here we use 96 values every quarter of an hour
    #standard load profile -- input
    q = extra.read_slp(t, 'Profielen-Elektriciteit-2015-versie-1.00 Folder/profielen Elektriciteit 2015 versie 1.00.csv') # read the sample standard load profile, can be any length, can be resized given a low/high resolution time axis
    q = q / np.sum(q) # normalization of standard load profile
    # process duration
    duration_axis = np.linspace(0.0, 24.0, 96.0)
    (p_d, E_p) = extra.app_time(duration_axis, 10, 2, 0.0, 24.0) # function that define the pdf of duration of a process
    # process consumption
    consumption_axis = np.linspace(0.0, 3.5, 96.0)
    (p_k, E_k) = extra.app_consumption(consumption_axis, 10, 2, 0.0, 3.5) # function that define the pdf of duration of a process
    # pdf of starting time
    p_t_0 = lpd.infer_t_0(q, p_d, E_k) # computes the pdf of starting time of processes
    p_t_0 = p_t_0 / np.sum(p_t_0) # normalization of the pdf to sum up to zero

    """
    1st Approach, starting time of processes is a discrete propapibility density function
    """
    # synthetic profile of D processes
    D = 2000
    synthetic_profile = lpd.synthetic_profile(D, t, p_d, consumption_axis, p_k, p_t_0)
    # expected value of D processes
    q_e_e = lpd.infer_q_e(t, p_t_0, p_d, E_k, D)
    # plot
    plt.step(t, synthetic_profile, "g-")
    plt.step(t, q_e_e, "b--")

    """
    2nd Approach, starting time of processes is a continuous propapibility density function
    """
    # synthetic profile of D processes
    ts, cs = lpd.continous_synthetic_profile(D, t, p_d, consumption_axis, p_k, p_t_0)
    plt.step(ts/len(t)*t[-1], cs, where='post', c='r')
    plt.xlim(0,24.0)
    plt.legend(["synthetic","expected", "continuous"],loc=0)
    plt.show()

    """
    Time discretization
    """
    n_intervals = 24*60 # discretized in minutes
    discrete_timeaxis = np.linspace(0.0, 24.0, n_intervals+1)
    discrete_consumption = lpd.signal_discretization(discrete_timeaxis, t, ts, cs)
    plt.step(ts/len(t)*t[-1], cs, where='post', c='r')
    plt.step(discrete_timeaxis, discrete_consumption, where='post', c='k', ls='--',lw=2)
    plt.legend(["continuous", "discretized"],loc=0)
    plt.show()


if __name__ == "__main__":
    main()
