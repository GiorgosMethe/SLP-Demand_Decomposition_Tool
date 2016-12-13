import lpd
import extra

import matplotlib.pyplot as plt

import numpy as np

np.random.seed()

def create_synthetic_load(load_profile, scale, days):

    t = np.linspace(0.0, 24.0, len(load_profile))
    load_profile /= np.sum(load_profile)

    duration_axis = np.linspace(0.0, 24.0, len(load_profile))
    (p_d, E_p) = extra.app_time(duration_axis, 10, 2, 0.0, 24.0)

    consumption_axis = np.linspace(0.0, 3.5, 100.0)
    (p_k, E_k) = extra.app_consumption(consumption_axis, 10, 2, 0.0, 3.5)

    p_t_0 = lpd.infer_t_0(load_profile, p_d, E_k)
    p_t_0 = p_t_0 / np.sum(p_t_0)

    q_e = lpd.infer_q_e(t, p_t_0, p_d, E_k, 1)

    D = int(len(load_profile) * scale / np.sum(q_e))

    return lpd.synthetic_profile_repeated(D, t, p_d, consumption_axis, p_k, p_t_0, days)


