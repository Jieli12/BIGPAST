"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2025-01-16 15:14:28
Last Edited : 2025-01-17 11:31:54
Last Author : Jie Li
File Path   : /BIGPAST/Python/sim_3_2_mixed_greater_transformation.py
Description :








Copyright (c) 2025, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %% import packages
import warnings

import array_to_latex as a2l
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t
from skewt_scipy.skewt import skewt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import *

# %% parameter  settings
warnings.filterwarnings("ignore")

alpha = -3.23
df = 7
loc = 0
scale = 1
x = np.linspace(-6, 6, 1000)
y = skewt.pdf(x, alpha, df, loc, scale)
plt.plot(x, y)
plt.show()
# %% generate data with fixed seed for reproducibility
np.random.seed(2024)
n_all = np.array([50, 100, 200, 400])
N_freq = 100
m = 10000
sig_level = 0.05
burn_in = 0.5
stepsize = np.array([0.05, 0.05, 0.05, 0.05])
cv = 1
each = 100
size = 2000
# Alternative: greater
# actual conditions,
# 0 or False: single case and control group comes from same distribution
# 1 or True: single case and control group comes from different distribution
actual_greater = np.repeat((False, True), N_freq * m // 2)
z_critical = norm.ppf(1 - sig_level, 0, 1)
t_critical = t.ppf(1 - sig_level, n_all - 1)

quantiles = np.array([sig_level / 2, 1 - sig_level / 2, sig_level, 1 - sig_level])
critical_values = skewt.ppf(quantiles, alpha, df, loc, scale)
critical_bounds = skewt.ppf([0.001, 0.999], alpha, df, loc, scale)
m2 = N_freq * m // 2

credible_int_bigpast = np.zeros((N_freq, len(n_all)))
bigpast_results = np.zeros((N_freq, len(n_all)))
cg_results = np.zeros((N_freq, len(n_all)))
ad_results = np.zeros((N_freq, len(n_all)))
results_all = np.full((len(n_all), 3, 1), np.nan)
bigpast_pred = np.zeros((N_freq * m, len(n_all)))
cg_pred = np.zeros((N_freq * m, len(n_all)))
ad_pred = np.zeros((N_freq * m, len(n_all)))

for j, (n, t_th) in enumerate(zip(n_all, t_critical)):
    data_control = skewt.rvs(alpha, df, loc, scale, size=(N_freq, n))
    d_s1_greater = np.random.uniform(critical_bounds[0], critical_values[3], m2)
    d_s2_greater = np.random.uniform(critical_values[3], critical_bounds[1], m2)
    data_single_case = np.concatenate([d_s1_greater, d_s2_greater]).reshape((N_freq, m))
    # Hyperbolic Arcsine transformation
    data_control_tran = np.log(data_control + np.sqrt(data_control**2 + 1))
    data_single_case_tran = np.log(data_single_case + np.sqrt(data_single_case**2 + 1))
    for i, row in tqdm(enumerate(data_control_tran)):
        ########### CG ##################
        # greater
        cg_p_value = np.zeros(m)
        cg_greater = np.zeros(m)
        d_greater = data_single_case_tran[i, :]
        for k in range(m):
            cg_p_value[k] = BTD(d_greater[k], row, alternative="greater")["p-value"]
        cg_pred[m * i : m * (i + 1), j] = cg_p_value <= sig_level

    conf_mat = confusion_matrix(actual_greater, cg_pred[:, j].reshape((N_freq * m,)))
    fpr_cg_greater = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_cg_greater = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_cg_greater = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_cg_greater, tpr_cg_greater, acc_cg_greater)
    results_all[j, :, 0] = [fpr_cg_greater, tpr_cg_greater, acc_cg_greater]


# %%
results_table = np.transpose(results_all, (1, 0, 2)).reshape((12, 1))
latex_code = a2l.to_ltx(
    results_table, frmt="{:5.4f}", arraytype="bmatrix", print_out=True
)

# %%
