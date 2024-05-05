"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-05 10:23:26
Last Revision : 2024-05-05 10:34:02
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_1_result.py
Description   : result of sim_3_1.py








Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import numpy as np
from utils import *

# %% set the parameters for simulation
alpha = [-1, -10, -30, -50, -1, -50]
df = [1, 10, 30, 50, 50, 1]
loc = -2
scale = np.sqrt(2)
N = 1000
n = 500
num_cases = len(alpha)
# %%
result_map = np.load("../Data/result_map.npy")
result_map_b = np.load("../Data/result_map_b.npy")
result_map_d = np.load("../Data/result_map_d.npy")
result_mle = np.load("../Data/result_mle.npy")

# %%
true_para = np.array(
    [alpha, df, np.repeat(loc, num_cases), np.repeat(scale, num_cases)]
).T
bias_0 = result_map[:, :, :, 0] - true_para
bias_1 = result_map[:, :, :, 1] - true_para
bias_b = result_map_b[:, :, :] - true_para
bias_d = result_map_d[:, :, :] - true_para
bias_m = result_mle[:, :, :] - true_para

l1_0 = np.nanmean(np.abs(bias_0), axis=0)
l1_1 = np.nanmean(np.abs(bias_1), axis=0)
l1_b = np.nanmean(np.abs(bias_b), axis=0)
l1_d = np.nanmean(np.abs(bias_d), axis=0)
l1_m = np.nanmean(np.abs(bias_m), axis=0)
result_com = np.zeros((20, 6))
for i in range(4):
    result_com[5 * i + 0, :] = l1_0[:, i]
    result_com[5 * i + 1, :] = l1_1[:, i]
    result_com[5 * i + 2, :] = l1_b[:, i]
    result_com[5 * i + 3, :] = l1_d[:, i]
    result_com[5 * i + 4, :] = l1_m[:, i]
print(np.array2string(result_com, suppress_small=True, precision=3))

# %%
