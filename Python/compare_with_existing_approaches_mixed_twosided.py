"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-04-19 21:08:46
Last Revision : 2024-04-19 21:44:18
Last Author   : Jie Li
File Path     : /bigpast_repo/Python/compare_with_existing_approaches_mixed_twosided.py
Description   : This script is used to compare the performance of the proposed BIGPAST against existing approaches: z-score, t-score (Crawford & Howell, 1998), Crawford- Garthwaite Bayesian approach (Crawford & Garthwaite, 2007), and Anderson-Darling non-parametric approach (Anderson & Darling, 1954).

The data are genereated using skew t distribution the severe skew settings of Section 3.1.1 in Crawford et al. (2006).

The single case observations consist of 50% positive and 50% negative. The direction of alternative hypothesis is `two-sided'.








Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
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
# Alternative: twosided
# actual conditions,
# 0 or False: single case and control group comes from same distribution
# 1 or True: single case and control group comes from different distribution
z_critical = norm.ppf((sig_level / 2, 1 - sig_level / 2), 0, 1)


quantiles = np.array([sig_level / 2, 1 - sig_level / 2, sig_level, 1 - sig_level])
critical_values = skewt.ppf(quantiles, alpha, df, loc, scale)
critical_bounds = skewt.ppf([0.001, 0.999], alpha, df, loc, scale)
m4 = N_freq * m // 4

credible_int_bigpast = np.zeros((N_freq, 2, len(n_all)))
bigpast_results = np.zeros((N_freq, len(n_all)))
cg_results = np.zeros((N_freq, len(n_all)))
ad_results = np.zeros((N_freq, len(n_all)))
results_all = np.full((len(n_all), 3, 5), np.nan)
bigpast_pred = np.zeros((N_freq * m, len(n_all)))
cg_pred = np.zeros((N_freq * m, len(n_all)))
ad_pred = np.zeros((N_freq * m, len(n_all)))

for j, n in enumerate(n_all):
    data_control = skewt.rvs(alpha, df, loc, scale, size=(N_freq, n))
    d_s1_twosided = np.random.uniform(critical_values[0], critical_values[1], m4 * 2)
    d_s2_twosided = np.random.uniform(critical_bounds[0], critical_values[0], m4)
    d_s3_twosided = np.random.uniform(critical_values[1], critical_bounds[1], m4)
    data_single_case = np.concatenate([d_s2_twosided, d_s1_twosided, d_s3_twosided])
    actual_twosided = np.logical_or(
        (data_single_case <= critical_values[0]),
        (data_single_case >= critical_values[1]),
    )
    data_single_case = data_single_case.reshape((N_freq, m))
    # z-score
    x_bar = np.mean(data_control, axis=1, keepdims=True)
    s = np.std(data_control, axis=1, keepdims=True)
    z_hat = (data_single_case - x_bar) / s
    z_hat_pred = np.logical_or((z_hat < z_critical[0]), (z_hat > z_critical[1]))
    conf_mat = confusion_matrix(actual_twosided, z_hat_pred.reshape((N_freq * m,)))
    fpr_z_twosided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_z_twosided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_z_twosided = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_z_twosided, tpr_z_twosided, acc_z_twosided)
    results_all[j, :, 0] = [fpr_z_twosided, tpr_z_twosided, acc_z_twosided]
    # t-score
    t_hat = z_hat * np.sqrt(n / (n + 1))
    t_critical = t.ppf((sig_level / 2, 1 - sig_level / 2), n - 1)
    t_hat_pred = np.logical_or((t_hat < t_critical[0]), (t_hat > t_critical[1]))
    conf_mat = confusion_matrix(actual_twosided, t_hat_pred.flatten())
    fpr_t_twosided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_t_twosided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_t_twosided = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_t_twosided, tpr_t_twosided, acc_t_twosided)
    results_all[j, :, 1] = [fpr_t_twosided, tpr_t_twosided, acc_t_twosided]

    for i, row in tqdm(enumerate(data_control)):
        ########### BIGPAST ##################
        alpha_init = np.random.uniform(0.9 * alpha, 1.1 * alpha, 1)
        df_init = np.random.uniform(0.9 * df, 1.1 * df, 1)
        loc_init = np.random.uniform(-0.1, 0.1, 1)
        scale_init = np.random.uniform(0.9 * scale, 1.1 * scale, 1)
        init_params = np.concatenate([alpha_init, df_init, loc_init, scale_init])
        # carry out the BIGPAST test
        sample_param = metropolis_hastings_sampling(
            init_params, row, size=size, burn_in=burn_in, stepsize=stepsize, cv=cv
        )
        para_bigpast = sample_param.T
        accept_rate = np.mean(np.diff(sample_param, axis=0) != 0)
        unqiue_rows, counts = np.unique(sample_param, axis=0, return_counts=True)
        sample_b = np.array([])
        for u_row, count in zip(unqiue_rows, counts):
            x_temp = skewt.rvs(
                u_row[0], u_row[1], u_row[2], u_row[3], size=count * each
            )
            sample_b = np.append(sample_b, x_temp)
        credible_int_bigpast[i, :, j] = np.quantile(
            sample_b, (sig_level / 2, 1 - sig_level / 2)
        )

        # twosided
        bigpast_twosided = np.logical_or(
            (data_single_case[i, :] >= credible_int_bigpast[i, 1, j]),
            (data_single_case[i, :] <= credible_int_bigpast[i, 0, j]),
        )
        bigpast_pred[m * i : m * (i + 1), j] = bigpast_twosided

        ########### CG ##################
        # twosided
        cg_p_value = np.zeros(m)
        cg_twosided = np.zeros(m)
        d_twosided = data_single_case[i, :]
        for k in range(m):
            cg_p_value[k] = BTD(d_twosided[k], row, alternative="two_sided")["p-value"]
        cg_pred[m * i : m * (i + 1), j] = cg_p_value <= sig_level

        ########### AD ##################
        # twosided
        ad = np.zeros(m)
        for k in range(m):
            res = stats.anderson_ksamp([row, d_twosided[k : k + 1]])
            ad[k] = res.pvalue <= sig_level
        ad_pred[m * i : m * (i + 1), j] = ad

    conf_mat = confusion_matrix(
        actual_twosided, bigpast_pred[:, j].reshape((N_freq * m,))
    )
    fpr_bigpast_twosided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_bigpast_twosided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_bigpast_twosided = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_bigpast_twosided, tpr_bigpast_twosided, acc_bigpast_twosided)
    results_all[j, :, 4] = [
        fpr_bigpast_twosided,
        tpr_bigpast_twosided,
        acc_bigpast_twosided,
    ]

    conf_mat = confusion_matrix(actual_twosided, cg_pred[:, j].reshape((N_freq * m,)))
    fpr_cg_twosided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_cg_twosided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_cg_twosided = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_cg_twosided, tpr_cg_twosided, acc_cg_twosided)
    results_all[j, :, 2] = [fpr_cg_twosided, tpr_cg_twosided, acc_cg_twosided]

    conf_mat = confusion_matrix(actual_twosided, ad_pred[:, j].reshape((N_freq * m,)))
    fpr_ad_twosided = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    tpr_ad_twosided = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    acc_ad_twosided = (conf_mat[0, 0] + conf_mat[1, 1]) / N_freq / m
    print(fpr_ad_twosided, tpr_ad_twosided, acc_ad_twosided)
    results_all[j, :, 3] = [fpr_ad_twosided, tpr_ad_twosided, acc_ad_twosided]

# %%
# save the results
np.save(
    f"../Data/bigpast_vs_others_mixed_twosided_N{N_freq}m{m}alpha{alpha}df{df}.npy",
    {
        "results_all": results_all,
        # "bigpast_results": bigpast_results,
        # "cg_results": cg_results,
        # "ad_results": ad_results,
    },
)

# %%
results_table = np.transpose(results_all, (1, 0, 2)).reshape((12, 5))
latex_code = a2l.to_ltx(
    results_table, frmt="{:7.6f}", arraytype="bmatrix", print_out=True
)

# %%
