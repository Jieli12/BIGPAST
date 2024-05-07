"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-07 14:42:32
Last Revision : 2024-05-07 14:47:18
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_4_plot.py
Description   :








Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import array_to_latex as a2l
import matplotlib.pyplot as plt
import numpy as np
from joblib import load

K = 2000
N = 400
n = 100
alpha = np.array([0.0, 1, 3, 5, 10, 20, 30, 50, 5, 50])
df = np.array([3.0, 3, 3, 5, 10, 20, 30, 50, 50, 5])
alternative = "two_sided"
avg_result = np.zeros((10, 8))
for j in range(10):
    fname = (
        f"../Data/Result_n{n}_alpha{alpha[j]}_df{df[j]}_alternative{alternative}.joblib"
    )
    results = load(fname)

    fdr = np.zeros((4, N))
    accuracy = np.zeros((4, N))
    for i, res_i in enumerate(results):
        confusion_matrix = res_i[0]
        fdr[:, i] = confusion_matrix[:, 1] / (
            confusion_matrix[:, 0] + confusion_matrix[:, 1]
        )
        accuracy[:, i] = (confusion_matrix[:, 0] + confusion_matrix[:, 3]) / K
    fdr[[1, 3]] = fdr[[3, 1]]
    accuracy[[1, 3]] = accuracy[[3, 1]]
    avg_result[j, :4] = np.mean(fdr.T, axis=0)
    avg_result[j, 4:] = np.mean(accuracy.T, axis=0)

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

    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
        ax.set_xlabel("Four separate methods")

    figname = f"../Data/Result_n{n}_alpha{int(alpha[j])}_df{int(df[j])}.pdf"
    plt.savefig(figname, format="pdf")


a2l.to_ltx(avg_result, frmt="{:4.3f}", arraytype="array")

# %%
