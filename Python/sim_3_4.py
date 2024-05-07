"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-07 12:00:40
Last Revision : 2024-05-07 14:29:37
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_4.py
Description   : This is a command line script which uses the parallel computation. The default number of CPU cores is 90. Please adjust the number of CPU cores according to your machine. The script generates the boxplot of FDR and accuracy for the four methods: MH, MAP, NP, and MLE. The script saves the boxplot in the `Data` folder. The script also saves the results in the joblib file in the Data folder.

Usages:

python bayes_procedure.py -a 10 -d 10 -n 100 -al two_sided




Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, dump
from skewt_scipy.skewt import skewt
from utils import *

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--alpha", type=float, dest="shape_parameter", help="shape parameter"
)
parser.add_argument("-n", "--length", type=int, dest="sample_size", help="sample size")
parser.add_argument(
    "-d", "--df", type=float, dest="degree_of_freedom", help="degree of freedom"
)
parser.add_argument(
    "-al", "--alternative", dest="alternative_hypothesis", help="alternative hypothesis"
)
args = parser.parse_args()
alpha = args.shape_parameter
df = args.degree_of_freedom
n = args.sample_size
alternative = args.alternative_hypothesis
# %% parameter settings
loc = -2
scale = 2
# %% sampling the single cases
np.random.seed(2024)
n1 = 1000
n2 = 1000
sig_level = 0.05

x_single, underlying = generate_single_subject(
    n1, n2, alpha, df, loc, scale, sig_level, alternative
)
# %%
N = 400
data = skewt.rvs(a=alpha, df=df, loc=loc, scale=scale, size=n * N)
data = np.reshape(data, (N, n))
init_params = np.array([alpha, df, loc, scale]) * 1.1
stepsize = [0.1, 0.1, 0.1, 0.1]
each = 100
# %% Parallel computation
results = Parallel(n_jobs=90)(
    delayed(compute_fdr_acc)(
        data_i, init_params, x_single, stepsize, underlying, sig_level, alternative
    )
    for data_i in data
)
fname = f"../Data/Result_n{n}_alpha{alpha}_df{df}_alternative{alternative}.joblib"
dump(results, fname)
fdr = np.zeros((4, N))
accuracy = np.zeros((4, N))
for i, res_i in enumerate(results):
    confusion_matrix = res_i[0]
    fdr[:, i] = confusion_matrix[:, 1] / (
        confusion_matrix[:, 0] + confusion_matrix[:, 1]
    )
    accuracy[:, i] = (confusion_matrix[:, 0] + confusion_matrix[:, 3]) / (n1 + n2)

# %%
labels = ["MH", "MAP", "NP", "MLE"]
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), dpi=600)
bplot1 = ax1.boxplot(fdr.T, vert=True, patch_artist=True, labels=labels)
ax1.set_title("FDR")
bplot2 = ax2.boxplot(accuracy.T, vert=True, patch_artist=True, labels=labels)
ax2.set_title("Accuracy")
# fill with colors
colors = ["pink", "lightblue", "lightgreen", "lightsalmon"]
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel("Four separate methods")

figname = f"../Data/Result_n{n}_alpha{alpha}_df{df}_alternative{alternative}.pdf"
plt.savefig(figname, format=".pdf")
