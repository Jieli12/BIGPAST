"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-05 09:55:47
Last Revision : 2024-05-05 10:22:30
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_1.py
Description   : This script is used to compare the performance of our prior against the existing priors in Section 3.1.








Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import numpy as np
from joblib import Parallel, delayed
from skewt_scipy.skewt import skewt
from tqdm import tqdm
from utils import *

# %% set the parameters for simulation
alpha = [-1, -10, -30, -50, -1, -50]
df = [1, 10, 30, 50, 50, 1]
loc = -2
scale = np.sqrt(2)
N = 1000
n = 500
np.random.seed(2024)
num_cases = len(alpha)
data_all = np.zeros((N, n, num_cases))
for i in range(N):
    for j in range(num_cases):
        data_all[i, :, j] = skewt.rvs(
            a=alpha[j], df=df[j], loc=loc, scale=scale, size=n
        )

# %%

result_map = np.zeros((N, num_cases, 4, 2))
result_map_b = np.zeros((N, num_cases, 4))
result_map_d = np.zeros((N, num_cases, 4))
result_mle = np.zeros((N, num_cases, 4))
results = Parallel(n_jobs=-1)(
    delayed(process_case)(data, alpha, df, loc, scale)
    for data in tqdm(data_all, desc="Processing", total=len(data_all))
)

for i, result in enumerate(results):
    (
        result_map[i, :, :, :],
        result_map_b[i, :, :],
        result_map_d[i, :, :],
        result_mle[i, :, :],
    ) = result


# %%
np.save("../Data/result_map.npy", result_map)
np.save("../Data/result_map_b.npy", result_map_b)
np.save("../Data/result_map_d.npy", result_map_d)
np.save("../Data/result_mle.npy", result_mle)

# %%
