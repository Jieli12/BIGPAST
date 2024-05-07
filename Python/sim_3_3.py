"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-07 09:51:40
Last Revision : 2024-05-07 10:13:51
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_3.py
Description   :

The parameter setting are: \alpha=3, \nu= 5






Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from skewt_scipy.skewt import skewt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import *

warnings.filterwarnings("ignore")

alpha = 3
df = 5
loc = 0
scale = 1
x = np.linspace(-6, 6, 1000)
y = skewt.pdf(x, alpha, df, loc, scale)
mu, variance = skewt.stats(alpha, df, loc, scale, moments="mv")

y1 = norm.pdf(x, mu, np.sqrt(variance))
plt.plot(x, y)
plt.plot(x, y1)
plt.show()
# %% generate data with fixed seed for reproducibility
np.random.seed(2024)
n = 100
N = 200
m4 = 250
m = 4 * m4
data_control = skewt.rvs(alpha, df, loc, scale, size=(N, n))
sig_level = 0.05

# Alternative: two-sided, less, greater
quantiles = np.array([sig_level / 2, 1 - sig_level / 2, sig_level, 1 - sig_level])
critical_values = skewt.ppf(quantiles, alpha, df, loc, scale)
critical_bounds = skewt.ppf([0.001, 0.999], alpha, df, loc, scale)
print(critical_values)
print(critical_bounds)
# generate single case data for two sided
d_s1_twosided = np.random.uniform(critical_values[0], critical_values[1], m4 * 2)
d_s2_twosided = np.random.uniform(critical_bounds[0], critical_values[0], m4)
d_s3_twosided = np.random.uniform(critical_values[1], critical_bounds[1], m4)
d_twosided = np.concatenate([d_s2_twosided, d_s1_twosided, d_s3_twosided])
# generate single case data for less
d_s1_less = np.random.uniform(critical_bounds[0], critical_values[2], m4 * 2)
d_s2_less = np.random.uniform(critical_values[2], critical_bounds[1], m4 * 2)
d_less = np.concatenate([d_s1_less, d_s2_less])
# generate single case data for greater
d_s1_great = np.random.uniform(critical_bounds[0], critical_values[3], m4 * 2)
d_s2_great = np.random.uniform(critical_values[3], critical_bounds[1], m4 * 2)
d_great = np.concatenate([d_s1_great, d_s2_great])


# actual conditions,
# 0: single case comes and control group comes from same distribution
# 1: single case comes and control group comes from different distribution

actual_two_sided = np.logical_or(
    (d_twosided <= critical_values[0]), (d_twosided >= critical_values[1])
)
actual_less = d_less <= critical_values[2]
actual_great = d_great >= critical_values[3]

# %%
burn_in = 0.5
stepsize = np.array([0.05, 0.05, 0.05, 0.05])
cv = 1
each = 100
size = 2000
credible_int_bigpast = np.zeros((N, 4))
bigpast_results = np.zeros((N, 9))
cg_results = np.zeros((N, 9))
ad_results = np.zeros((N, 9))
for i, row in tqdm(enumerate(data_control)):
    ########### BIGPAST ##################
    alpha_init = np.random.uniform(0.9 * alpha, 1.1 * alpha, 1)
    df_init = np.random.uniform(0.9 * df, 1.1 * df, 1)
    loc_init = np.random.uniform(-0.1, 0.1, 1)
    scale_init = np.random.uniform(0.9 * scale, 1.1 * scale, 1)
    init_params = np.concatenate([alpha_init, df_init, loc_init, scale_init])
    # carry out the BIGPAST test
    sample_param = metropolis_hastings(
        init_params, row, size=size, burn_in=burn_in, stepsize=stepsize, cv=cv
    )
    para_bigpast = sample_param.T
    accept_rate = np.mean(np.diff(sample_param, axis=0) != 0)
    unqiue_rows, counts = np.unique(sample_param, axis=0, return_counts=True)
    sample_b = np.array([])
    for u_row, count in zip(unqiue_rows, counts):
        x_temp = skewt.rvs(u_row[0], u_row[1], u_row[2], u_row[3], size=count * each)
        sample_b = np.append(sample_b, x_temp)
    credible_int_bigpast[i, :] = np.quantile(sample_b, quantiles)
    # calculate the type I error (false positive rate) and power (true positive rate) of bigpast
    # two-sided
    bigpast_two_sided = np.logical_or(
        d_twosided <= credible_int_bigpast[i, 0],
        d_twosided >= credible_int_bigpast[i, 1],
    )
    conf_mat = confusion_matrix(actual_two_sided, bigpast_two_sided)
    fpr_two_sided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_two_sided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_two_sided = (conf_mat[0, 0] + conf_mat[1, 1]) / m

    # less
    bigpast_less = d_less <= credible_int_bigpast[i, 2]
    conf_mat = confusion_matrix(actual_less, bigpast_less)
    fpr_less = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_less = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_less = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    # great
    bigpast_great = d_great >= credible_int_bigpast[i, 3]
    conf_mat = confusion_matrix(actual_great, bigpast_great)
    fpr_great = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_great = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_great = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    bigpast_fpr_tpr = np.array(
        [
            fpr_two_sided,
            tpr_two_sided,
            acc_two_sided,
            fpr_less,
            tpr_less,
            acc_less,
            fpr_great,
            tpr_great,
            acc_great,
        ]
    )
    ########### CG ##################
    # two-sided
    cg_two_sided = np.zeros(m)
    for j in range(m):
        cg_two_sided[j] = (
            BTD(d_twosided[j], row, alternative="two_sided")["p-value"] <= sig_level
        )
    conf_mat = confusion_matrix(actual_two_sided, cg_two_sided)
    fpr_two_sided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_two_sided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_two_sided = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    # less
    cg_p_value = np.zeros(m)
    cg_less = np.zeros(m)
    for j in range(m):
        cg_p_value[j] = BTD(d_less[j], row, alternative="less")["p-value"]
    conf_mat = confusion_matrix(actual_less, cg_p_value <= sig_level)
    fpr_less = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_less = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_less = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    # great
    cg_great = np.zeros(m)
    for j in range(m):
        cg_great[j] = (
            BTD(d_great[j], row, alternative="greater")["p-value"] <= sig_level
        )
    conf_mat = confusion_matrix(actual_great, cg_great)
    fpr_great = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_great = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_great = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    cg_fpr_tpr = np.array(
        [
            fpr_two_sided,
            tpr_two_sided,
            acc_two_sided,
            fpr_less,
            tpr_less,
            acc_less,
            fpr_great,
            tpr_great,
            acc_great,
        ]
    )
    ########### AD ##################
    # two-sided
    ad = np.zeros(m)
    for j in range(m):
        res = stats.anderson_ksamp([row, d_twosided[j : j + 1]])
        ad[j] = res.pvalue <= sig_level
    conf_mat = confusion_matrix(actual_two_sided, ad)
    fpr_two_sided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_two_sided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_two_sided = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    # less
    conf_mat = confusion_matrix(actual_less, ad)
    fpr_less = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_less = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_less = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    # great
    conf_mat = confusion_matrix(actual_great, ad)
    fpr_less = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_less = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_great = (conf_mat[0, 0] + conf_mat[1, 1]) / m
    ad_fpr_tpr = np.array(
        [
            fpr_two_sided,
            tpr_two_sided,
            acc_two_sided,
            fpr_less,
            tpr_less,
            acc_less,
            fpr_great,
            tpr_great,
            acc_great,
        ]
    )
    bigpast_results[i, :] = bigpast_fpr_tpr
    cg_results[i, :] = cg_fpr_tpr
    ad_results[i, :] = ad_fpr_tpr
# %% save the results
np.save(
    f"../Data/bigpast_vs_cgn{n}N{N}m{m}alpha{alpha}df{df}.npy",
    {
        "bigpast_results": bigpast_results,
        "cg_results": cg_results,
        "ad_results": ad_results,
    },
)
# %%
