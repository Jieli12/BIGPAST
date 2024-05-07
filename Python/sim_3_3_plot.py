"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-05-07 10:08:02
Last Revision : 2024-05-07 10:14:58
Last Author   : Jie Li
File Path     : /BIGPAST/Python/sim_3_3_plot.py
Description   :








Copyright (c) 2024, Jie Li, jie.li@innovision-ip.co.uk and jl725@kent.ac.uk
All Rights Reserved.
"""

# %%

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import quad
from scipy.stats import norm
from skewt_scipy.skewt import skewt
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

params = [alpha, df, loc, scale, mu, np.sqrt(variance)]


def tv_distance(xx, params):
    alpha, df, loc, scale, mu, sigma = params
    return 0.5 * np.abs(skewt.pdf(xx, alpha, df, loc, scale) - norm.pdf(xx, mu, sigma))


func_tv = lambda x: tv_distance(x, params)
dist_tv = quad(func_tv, -np.inf, np.inf)[0]

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(
    x,
    y,
    linewidth=3,
    label=f"Skew $t$ distribution: \n $ \\alpha $={alpha},  $\\nu$={df}\n $\\xi$={loc},  $\\omega$={scale}",
)
plt.plot(
    x,
    y1,
    linewidth=3,
    label=f"Normal distribution: \n$\\mu$={mu:.2f}, $\\sigma^2$={variance:.2f}",
)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("Density", fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
legend = plt.legend(fontsize=14)
ax.text(
    0.027,
    0.6,
    f"TV distance: {dist_tv:.3f}",
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)
fpdf = f"../figures/alpha{alpha}df{df}.pdf"
fig.savefig(fpdf, format="pdf")
plt.show()
# %%
np.random.seed(2024)
n = 100
N = 200
m4 = 250
m = 4 * m4
sig_level = 0.05

# %%
data = np.load(
    f"../Data/bigpast_vs_cgn{n}N{N}m{m}alpha{alpha}df{df}.npy", allow_pickle=True
).item()

bigpast_results = data["bigpast_results"]
cg_results = data["cg_results"]
ad_results = data["ad_results"]
# plot the results
FPR = np.concatenate(
    [
        bigpast_results[:, 0],
        cg_results[:, 0],
        ad_results[:, 0],
        bigpast_results[:, 3],
        cg_results[:, 3],
        np.repeat(np.nan, N),
        bigpast_results[:, 6],
        cg_results[:, 6],
        np.repeat(np.nan, N),
    ]
)
approach0 = np.repeat(["BIGPAST", "CG", "AD"], N)
Approach = np.tile(approach0, 3)
Alternative = np.repeat(["two-sided", "less", "greater"], N * 3)

type1 = pd.DataFrame(
    {"Type I error": FPR, "Approach": Approach, "Alternative": Alternative}
)

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.boxplot(
    data=type1,
    y="Type I error",
    x="Alternative",
    hue="Approach",
    fill=True,
    linewidth=2,
)
ax.set_xlabel("Alternative", fontsize=20)
ax.set_ylabel("Type I error", fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
plt.legend(fontsize=14)
fpdf = f"../figures/TypeIErroralpha{alpha}df{df}.pdf"
fig.savefig(fpdf, format="pdf")
plt.show()
# %%
TPR = np.concatenate(
    [
        bigpast_results[:, 1],
        cg_results[:, 1],
        ad_results[:, 1],
        bigpast_results[:, 4],
        cg_results[:, 4],
        np.repeat(np.nan, N),
        bigpast_results[:, 7],
        cg_results[:, 7],
        np.repeat(np.nan, N),
    ]
)
approach0 = np.repeat(["BIGPAST", "CG", "AD"], N)
Approach = np.tile(approach0, 3)
Alternative = np.repeat(["two-sided", "less", "greater"], N * 3)

power = pd.DataFrame({"Power": TPR, "Approach": Approach, "Alternative": Alternative})

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.boxplot(
    data=power, y="Power", x="Alternative", hue="Approach", fill=True, linewidth=2
)
ax.set_xlabel("Alternative", fontsize=20)
ax.set_ylabel("Power", fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
plt.legend(fontsize=14)
fpdf = f"../figures/Poweralpha{alpha}df{df}.pdf"
fig.savefig(fpdf, format="pdf")
plt.show()
# %%
ACC = np.concatenate(
    [
        bigpast_results[:, 2],
        cg_results[:, 2],
        ad_results[:, 2],
        bigpast_results[:, 5],
        cg_results[:, 5],
        np.repeat(np.nan, N),
        bigpast_results[:, 8],
        cg_results[:, 8],
        np.repeat(np.nan, N),
    ]
)
approach0 = np.repeat(["BIGPAST", "CG", "AD"], N)
Approach = np.tile(approach0, 3)
Alternative = np.repeat(["two-sided", "less", "greater"], N * 3)

acc = pd.DataFrame({"Accuracy": ACC, "Approach": Approach, "Alternative": Alternative})

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.boxplot(
    data=acc, y="Accuracy", x="Alternative", hue="Approach", fill=True, linewidth=2
)
ax.set_xlabel("Alternative", fontsize=20)
ax.set_ylabel("Accuracy", fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
plt.legend(fontsize=14)
fpdf = f"../figures/Accuracyalpha{alpha}df{df}.pdf"
fig.savefig(fpdf, format="pdf")
plt.show()
# %%
