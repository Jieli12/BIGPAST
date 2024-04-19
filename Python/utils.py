"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-04-18 14:40:57
Last Revision : 2024-04-18 20:59:08
Last Author   : Jie Li
File Path     : /bigpast_repo/Python/utils.py
Description   :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

# %% import packages

import numpy as np
import scipy.stats as stats
from scipy.optimize import Bounds, basinhopping, minimize
from scipy.special import beta, gamma, hyp2f1, loggamma, polygamma
from scipy.stats import norm
from skewt_scipy.skewt import skewt


# %% functions
def metropolis_hastings_sampling(
    init_params, data, size=10000, burn_in=0.4, stepsize=0.5, cv=1
):
    """obtain Metropolis Hasting's samples

    Parameters
    ----------
    init_params : array
        the initial parameters of the skew-t distribution in order: alpha, df, loc, scale
    data : array
        the input data
    size : int, optional
        the sample size, by default 10000
    burn_in : float, optional
        rate of burn_in, by default 0.4
    stepsize : float, optional
        the step size of parameters, by default 0.5
    cv : float, optional
        a constant depending on the degree of freedom, by default 1

    Returns
    -------
    2D array
        the matrix of samples with shape (size*(1-burn_in), num_params)
    """
    num_params = len(init_params)
    samples = np.zeros((size, num_params))
    params_curr = init_params
    stepsize = np.array(stepsize)

    if stepsize.size == 4:
        a_stepsize, df_stepsize, loc_stepsize, scale_stepsize = stepsize
    else:
        a_stepsize = df_stepsize = loc_stepsize = scale_stepsize = stepsize
    i = 0
    while i < size:
        a_curr, df_curr, loc_curr, scale_curr = params_curr
        df_new = df_curr + df_stepsize * norm.ppf(
            norm.cdf(-df_curr / df_stepsize)
            + np.random.rand() * norm.cdf(df_curr / df_stepsize)
        )
        p_df = norm.logcdf(df_curr / df_stepsize) - norm.logcdf(df_new / df_stepsize)
        scale_new = scale_curr + scale_stepsize * norm.ppf(
            norm.cdf(-scale_curr / scale_stepsize)
            + np.random.rand() * norm.cdf(scale_curr / scale_stepsize)
        )
        p_scale = norm.logcdf(scale_curr / scale_stepsize) - norm.logcdf(
            scale_new / scale_stepsize
        )
        a_new = a_curr + np.random.normal(0, a_stepsize, 1)
        loc_new = loc_curr + np.random.normal(0, loc_stepsize, 1)
        params_new = np.array([a_new[0], df_new, loc_new[0], scale_new])
        prob_new = -neg_log_posterior(params_new, data, cv=cv)
        prob_curr = -neg_log_posterior(params_curr, data, cv=cv)
        A = prob_new - prob_curr + p_df + p_scale
        if np.log(np.random.rand()) <= A:
            params_curr = params_new
        samples[i, :] = params_curr.flatten()
        i += 1

    return samples[int(size * burn_in) :, :]


def neg_log_posterior(params, data, cv=1):
    """compute the negative log posterior of the skew-t distribution

    Parameters
    ----------
    params : array
        the parameters of the skew-t distribution in order: alpha, df, loc, scale
    data : array
        the input data
    cv : float, optional
        a constant depending on the degree of freedom, by default 1

    Returns
    -------
    float
        the negative log posterior distribution
    """
    alpha, df, loc, scale = params
    return (
        -np.log(compute_fisher_det(alpha, df, cv))
        - log_likelihood_standard(alpha, df, loc, scale, data)
        + np.log(scale)
    )


def compute_fisher_det(alpha, v, cv):
    """compute the determinant of the Fisher information matrix

    Parameters
    ----------
    alpha : float
        the skewness parameter
    v : integer
        the degree of freedom
    cv : positive float
        a constant depending on the degree of freedom

    Returns
    -------
    float
        the determinant of the Fisher information matrix
    """
    s2 = (compute_sigma_nu(v + 1)) ** 2
    psi1 = polygamma(0, v / 2)
    psi2 = polygamma(0, v / 2 + 1 / 2)
    psi3 = polygamma(0, v / 2 + 1)
    g1 = -0.25 * (psi3 - psi2)
    dv = -0.5 * psi1 + 0.5 * psi3 + 2 * cv * g1
    if alpha == 0:
        i11 = np.pi * (gamma(v / 2 + 1)) ** 2 / (1 + v) / s2 / (gamma((v + 1) / 2)) ** 2
        i12 = 0
        i22 = (
            0.25
            * ((psi1 - psi2) ** 2 + polygamma(1, v / 2) - polygamma(1, (v + 1) / 2))
            + 1 / (v**2 + 3 * v) / 4
            + (dv**2)
            - 0.5 / (v**2 + v)
            + dv * (psi1 - psi2)
        )
    else:
        alpha2sigma = -(alpha**2) / s2
        v = np.round(v, 6)
        H1 = hyp2f1(0.5, (v + 1), (v + 3) / 2, alpha2sigma)
        H2 = hyp2f1(0.5, (v + 2), (v + 3) / 2, alpha2sigma)
        H3 = hyp2f1(1.5, (v + 1), (v + 5) / 2, alpha2sigma)
        H4 = hyp2f1(1.5, (v + 2), (v + 5) / 2, alpha2sigma)
        H5 = hyp2f1(-0.5, (v + 2), (v + 5) / 2, alpha2sigma)
        H6 = hyp2f1(0.5, (v + 2), (v + 5) / 2, alpha2sigma)
        H7 = hyp2f1(0.5, (v + 1), (v + 1) / 2, alpha2sigma)
        H8 = hyp2f1(0.5, (v + 2), (v + 1) / 2, alpha2sigma)

        ga1 = loggamma(v / 2 + 1 / 2)
        ga2 = loggamma(v / 2 + 1)
        ga5 = loggamma((v + 5) / 2)

        i12 = -(
            np.pi
            * alpha
            * np.exp(2 * ga2 - ga1 - ga5)
            * (H5 * (v + 4) - (alpha**2 * (2 * v + 3) + (v + 3) * s2) / s2 * H6)
            / 8
            / (alpha**2 + s2)
        )
        i11 = np.pi * np.exp(2 * ga2 - 2 * ga1) * (H7 - H8) / alpha**2
        i221 = (
            np.pi ** (5 / 2)
            * np.exp(ga2 - ga5)
            * ((v + 3) * (H1 - H2) + H4 - H3)
            / 8
            / v**2
            / beta(v / 2, 1 / 2)
            / (beta((v + 1) / 2, 1 / 2)) ** 2
        )

        i222 = (
            0.25
            * ((psi1 - psi2) ** 2 + polygamma(1, v / 2) - polygamma(1, (v + 1) / 2))
            + 1 / (v**2 + 3 * v) / 4
            + dv**2
            - 0.5 / (v**2 + v)
            + dv * (psi1 - psi2)
        )
        i22 = i221 + i222
    return np.sqrt(np.abs(i11 * i22 - i12**2))


def log_likelihood_standard(alpha, df, loc, scale, data):
    """compute the log likelihood of the skew-t distribution

    Parameters
    ----------
    alpha : float
        the skewness parameter
    df : integer
        the degree of freedom
    loc : float
        the location parameter
    scale : float
        the scale
    data : array
        the input data

    Returns
    -------
    float
        the log likelihood
    """
    return np.sum(skewt.logpdf(data, a=alpha, df=df, loc=loc, scale=scale))


def compute_sigma_nu(v):
    """compute the sigma_nu in appendix

    Parameters
    ----------
    v : integer
        the degree of freedom

    Returns
    -------
    float
        the sigma_nu
    """
    if v > 2400:
        return 1.5536
    else:
        log_v = np.log(v)
        return (
            0.00000543 * log_v**7
            - 0.00016303 * log_v**6
            + 0.00199613 * log_v**5
            - 0.01285016 * log_v**4
            + 0.04631303 * log_v**3
            - 0.08761023 * log_v**2
            + 0.05036188 * log_v
            + 1.62021189
        )


def BTD(case, controls, sd=None, sample_size=None, alternative="less", n_iter=10000):
    """Compute the Crawford Garthwaite test. This code is translated from R function BTD from R package `singcar'.

    Parameters
    ----------
    case : scalar
        the mean of test subject
    controls : 1D array
        the control sample
    sd : scalar, optional
        the standard deviation of the control sample, by default None
    sample_size : int, optional
        the  sample size of the control sample, by default None
    alternative : str, optional
        the alternative hypothesis, by default "less"
    n_iter : int, optional
        the number of iterations, by default 10000

    Returns
    -------
    dict
        the p value, ZCC score  and alternative

    """
    if len(controls) < 2 and sd is None:
        raise ValueError(
            "Not enough obs. Set sd and n for input of controls to be treated as mean"
        )

    if len(controls) < 2 and sd is not None and sample_size is not None:
        raise ValueError("Input sample size")
    if case is None:
        raise ValueError("Case is NA")

    con_m = np.nanmean(controls)  # Mean of the control sample
    con_sd = np.nanstd(controls)
    if len(controls) < 2 and sd is not None:
        con_sd = sd

    n = len(controls)
    if len(controls) < 2 and sd is not None and sample_size is not None:
        n = sample_size

    df = n - 1  # The degrees of freedom for chi2 simulation
    theta_hat = (n - 1) * con_sd**2 / np.random.chisquare(df, size=n_iter)
    z = np.random.normal(size=n_iter)
    mu_hat = con_m + z * np.sqrt(theta_hat / n)
    z_ast = (case - mu_hat) / np.sqrt(theta_hat)

    if alternative == "less":
        pval = stats.norm.cdf(z_ast)
    elif alternative == "greater":
        pval = 1 - stats.norm.cdf(z_ast)
    elif alternative == "two_sided":
        pval = 2 * (1 - stats.norm.cdf(abs(z_ast)))

    zcc = (case - con_m) / con_sd
    p_est = np.mean(pval)

    return {
        "p-value": p_est,
        "ZCC score": zcc,
        "alternative": alternative,
        "mu_hat": mu_hat,
        "theta_hat": theta_hat,
    }
