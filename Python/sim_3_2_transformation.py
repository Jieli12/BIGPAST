"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2025-01-16 10:23:45
Last Edited : 2025-01-17 17:07:44
Last Author : Jie Li
File Path   : /BIGPAST/Python/sim_3_2_transformation.py
Description : For the transformed data








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
# Alternative: less
# actual conditions,
# 0 or False: single case and control group comes from same distribution
# 1 or True: single case and control group comes from different distribution
actual_less = np.repeat(False, N_freq * m)
z_critical = norm.ppf(sig_level, 0, 1)
t_critical = t.ppf(sig_level, n_all - 1)

credible_int_bigpast = np.zeros((N_freq, len(n_all)))
bigpast_results = np.zeros((N_freq, len(n_all)))
cg_results = np.zeros((N_freq, len(n_all)))
ad_results = np.zeros((N_freq, len(n_all)))
results_all = np.full((len(n_all), 3, 1), np.nan)

for j, (n, t_th) in enumerate(zip(n_all, t_critical)):
    data_control = skewt.rvs(alpha, df, loc, scale, size=(N_freq, n))
    data_single_case = skewt.rvs(alpha, df, loc, scale, size=(N_freq, m))
    # Hyperbolic Arcsine transformation
    data_control_tran = np.log(data_control + np.sqrt(data_control**2 + 1))
    data_single_case_tran = np.log(data_single_case + np.sqrt(data_single_case**2 + 1))
    for i, row in tqdm(enumerate(data_control_tran)):

        ########### CG ##################
        # less
        cg_p_value = np.zeros(m)
        cg_less = np.zeros(m)
        d_less = data_single_case_tran[i, :]
        for k in range(m):
            cg_p_value[k] = BTD(d_less[k], row, alternative="less")["p-value"]
        cg_fpr = np.sum(cg_p_value <= sig_level) / m

        cg_results[i, j] = cg_fpr

results_all[:, 0, 0] = np.mean(cg_results, axis=0)
results_all[:, 2, 0] = 1 - np.mean(cg_results, axis=0)


# %%
results_table = np.transpose(results_all, (1, 0, 2)).reshape((12, 1))
latex_code = a2l.to_ltx(
    results_table, frmt="{:5.4f}", arraytype="bmatrix", print_out=True
)

# %%
