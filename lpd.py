import numpy as np
from operator import itemgetter

"""
Random samples method
Returns:
samples given a pdf
"""
def random_sample(distribution, size = 1):
    cdf = np.cumsum(distribution)
    random_ = np.random.uniform(size=size)
    samples = [np.where(cdf >= ran)[0][0] for ran in random_]
    return samples

"""
Random continous samples method
Returns:
continuous samples given a discrete pdf
"""
def random_continous_sample(distribution, axis=None, size = 1):
    continuous_samples = []
    cdf = np.cumsum(distribution)
    random_ = np.random.uniform(size=size)
    samples = [np.where(cdf >= ran)[0][0] for ran in random_]
    for sample in zip(samples,random_):
        if sample[0] == 0:continuous_samples.append(float(sample[0]))
        else:
            ratio = (sample[1] - cdf[sample[0]-1]) / (cdf[sample[0]] - cdf[sample[0]-1])

            if axis is not None:
                continuous_samples.append(axis[sample[0]-1] + (ratio * abs(axis[sample[0]] - axis[sample[0]-1])))
            else:
                continuous_samples.append(float(sample[0]) - (1.0 - ratio))
    return continuous_samples

"""
Linear interpolation upsampling
"""
def upsample(signal, new_signal):
    for i in range(len(signal)):
        new_pos = int(round((i)*(float(len(new_signal)-1) / float(len(signal) -1))))
        new_signal[new_pos] = signal[i]
        if (i > 0):
            prev_pos = int(round((i-1)*(float(len(new_signal)-1) / float(len(signal) -1))))
            for j in range(1, new_pos-prev_pos):
                new_prop = float(j) / (new_pos - prev_pos)
                prev_prop = 1.0 - new_prop
                new_signal[prev_pos + j] = (new_prop * new_signal[new_pos]) + (prev_prop * new_signal[prev_pos])
    return new_signal


"""
Infers and return the expected quantity, given the probability of starting time of a process,
the duration, and the expected value of process' consumption rate.
Returns:
expected quantity
"""
def infer_q_e(t, p_t_0, p_d, E_k = 1.0, D = 1.0):

    P_bar_d = np.zeros(len(p_d)) # cumulative complementary distribution function
    for i in range(len(p_d)):
        P_bar_d[i] = np.sum(p_d[i:])

    q_e = np.zeros(len(t))
    for i in range(len(t)):
        sum_td = 0.0
        for j in range(len(t)):
            sum_td = sum_td + (p_t_0[j] * P_bar_d[i-j])
        q_e[i] = float(D) * E_k * sum_td
    return q_e


"""
Infers the starting time probability density function of a process

Solves the linear system of equation A x t_0 = B

A[0,0] = probability of starting process at timestep zero and have more than zero duration
A[0,1] = probability of starting process at timestep zero and have more than one duration
...
A[1,0] = probability of starting process at timestep one and have more than n duration
A[1,1] = probability of starting process at timestep zero and have more than zero duration

t_0 = pdf of starting time of process

B = standard load profile

Returns:
PDF of starting time of process
"""
def infer_t_0(q, p_d, E_k):

    P_bar_d = np.zeros(len(p_d)) # cumulative complementary distribution function
    for i in range(len(p_d)):
        P_bar_d[i] = np.sum(p_d[i:])

    A = np.array([])
    B = np.ones(len(q))
    for i in range(len(q)):
        row = np.ones(len(p_d))
        for j in range(len(p_d)):
            row[j] = P_bar_d[i-j]
        if len(A) != 0:A = np.vstack((A,row))
        else:A = row
        B[i] = q[i]
    x  =  np.linalg.lstsq(A, B)[0]
    return x

"""
Constructs a synthetic profile
Returns:
Synthetic profile
"""
def synthetic_profile(D, t, d, consumption, k, t_0):
    ds = random_sample(d, D)
    ks = random_sample(k, D)
    t_0s = random_sample(t_0, D)

    slp = np.zeros(len(t))
    for d in zip(ds, consumption[ks], t_0s):
        for time in range(d[2], d[2]+d[0]+1): # +1 because range(0,0) = ~, range(0,1) = 0
            if (time >= len(t)):
                slp[time - len(t)] = slp[time - len(t)] + d[1]
            else:
                slp[time] = slp[time] + d[1]
    return slp

"""
Constructs a continuous synthetic profile
Returns:
tuples (time, value)
"""
def continous_synthetic_profile(D, t, d, consumption, k, t_0):

    t_0s = random_continous_sample(t_0, None, D)
    ds = random_continous_sample(d, None, D)
    ks = random_continous_sample(k, consumption, D)

    slp = list()
    for d in zip(ds, ks, t_0s): slp.append((d[2], d[2]+d[0]+2.0, d[1]))

    up = [(p[0], p[2]) for p in slp]
    down = [(p[1]%len(t), -p[2]) for p in slp]
    carryover = sum([p[2] for p in slp if p[1]>len(t)]) # these processes last more than the end of the day
    steps = up + down # append the two arrays
    steps.append((0, carryover)) #first timestep carries the value of all processes not ended during the day
    steps.append((len(t), 0)) #last timestep
    ssteps = sorted(steps, key=itemgetter(0)) # sorted by starting time
    matrix = np.array(ssteps)
    ts = matrix[:, 0] # timesteps
    cs = np.cumsum(matrix[:, 1]) # cumulative sum of all processes signals

    return ts, cs


"""
Going from a continuous consumption process to a discrete
Returns:
Discrete time signal
"""
def signal_discretization(timeaxis, t, ts, cs):
    discrete = np.zeros(len(timeaxis))
    v = zip(ts/(ts[-1]/timeaxis[-1]), cs)
    last_time, last_value = 0.0, 0.0
    index = 0
    for i in range(1, len(timeaxis)):
        if v[index][0] > timeaxis[i]: # process starts after this interval
            discrete[i-1] = discrete[i-1] + (last_value * (timeaxis[i] - timeaxis[i-1]))
        else:
            while v[index][0] <= timeaxis[i]:
                discrete[i-1] = discrete[i-1] + (last_value * (v[index][0] - max(last_time, timeaxis[i-1])))
                last_time, last_value = v[index][0], v[index][1]
                index = index + 1
                if index >= len(ts):break
            if index >= len(ts):break
            if v[index][0] > timeaxis[i]:
                discrete[i-1] = discrete[i-1] + (last_value * (timeaxis[i] - last_time))

    return discrete / (t[-1] / (len(timeaxis) - 1))
