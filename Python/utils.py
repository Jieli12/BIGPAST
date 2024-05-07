"""
Author        : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date          : 2024-04-18 14:40:57
Last Revision : 2024-05-07 12:40:17
Last Author   : Jie Li
File Path     : /BIGPAST/Python/utils.py
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


def process_case(data, alpha, df, loc, scale):
    """obtain the results of simulation using different priors of the skew-t distribution

    Parameters
    ----------
    data : float
        2D array of data
    alpha : float
        the skewness parameter
    df : positive float
        the degree of freedom
    loc : float
        the location parameter
    scale : float
        the scale parameter

    Returns
    -------
    dict
        the result of simulation using different priors
    """
    num_cases = data.shape[1]
    result_map = np.zeros((num_cases, 4, 2))
    result_map_b = np.zeros((num_cases, 4))
    result_map_d = np.zeros((num_cases, 4))
    result_mle = np.zeros((num_cases, 4))
    for j in range(num_cases):
        init_params = [alpha[j] * 1.1, df[j] * 1.1, loc * 1.1, scale * 1.1]
        data_j = data[:, j]
        try:
            result_map[j, :, 0] = map_posterior(data_j, 0, init_params)
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            result_map[j, :, 1] = map_posterior(data_j, 1, init_params)
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            result_map_b[j, :] = map_posterior_b(data_j, init_params)
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            result_map_d[j, :] = map_posterior_d(data_j, init_params)
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            result_mle[j, :] = skewt_fit(data_j, init_params)
        except Exception as e:
            print(f"An error occurred: {e}")
    return result_map, result_map_b, result_map_d, result_mle


def map_posterior(data, cv, init_params=np.array([1, 1, 1, 1])):
    bounds = Bounds([-np.inf, 1e-4, -np.inf, 1e-4], [np.inf, 1e8, np.inf, 1e8])
    res = minimize(
        fun=neg_log_posterior,
        x0=init_params,
        args=(data, cv),
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-10},
    )
    if res.success:
        fitted_params = res.x
        return fitted_params
    else:
        return np.repeat(np.nan, 4)


def map_posterior_d(data, init_params=np.array([1, 1, 1, 1])):
    bounds = Bounds([-np.inf, 1e-4, -np.inf, 1e-4], [np.inf, 1e8, np.inf, 1e8])
    res = minimize(
        fun=neg_log_posterior_d,
        x0=init_params,
        args=(data,),
        bounds=bounds,
        method="L-BFGS-B",
    )
    if res.success:
        fitted_params = res.x
        return fitted_params
    else:
        raise ValueError(res.message)


def map_posterior_b(data, init_params=np.array([1, 1, 1, 1])):
    bounds = Bounds([-np.inf, 1e-4, -np.inf, 1e-4], [np.inf, 1e8, np.inf, 1e8])
    res = minimize(
        fun=neg_log_posterior_b,
        x0=init_params,
        args=(data,),
        bounds=bounds,
        method="L-BFGS-B",
    )
    if res.success:
        fitted_params = res.x
        return fitted_params
    else:
        raise ValueError(res.message)


def neg_log_posterior_b(params, data):
    alpha, df, loc, scale = params
    return (
        -nu_prior_log(df)
        - alpha_prior_log_given_nu_t(alpha)
        - log_likelihood_standard(alpha, df, loc, scale, data)
        + np.log(scale)
    )


def alpha_prior_log_given_nu_t(alpha):
    return -0.75 * np.log(np.pi**2 + 8 * alpha**2)


def neg_log_posterior_d(params, data):
    alpha, df, loc, scale = params
    return (
        -nu_prior_log(df)
        + np.log(np.pi * (1 + alpha**2))
        - log_likelihood_standard(alpha, df, loc, scale, data)
        + np.log(scale)
    )


def nu_prior_log(v):
    if v <= 0:
        raise ValueError("v must be greater than 0")
    a = (v / (v + 3)) ** (1 / 2) * (
        polygamma(1, v / 2) - polygamma(1, (v + 1) / 2) - 2 * (v + 3) / v / (v + 1) ** 2
    ) ** (1 / 2)
    return np.log(a)


def alpha_prior_log_given_nu(alpha, v):
    sigma = compute_sigma_nu(v + 1)
    if v <= 0:
        raise ValueError("v must be greater than 0")
    a1 = (
        0.5 * np.log(np.pi)
        + loggamma(v / 2 + 1)
        - np.log(np.abs(alpha))
        - loggamma((v + 1) / 2)
    )
    a2 = 0.5 * np.log(
        hyp2f1(1 / 2, v + 1, (v + 1) / 2, -(alpha**2) / sigma**2)
        - hyp2f1(1 / 2, v + 2, (v + 1) / 2, -(alpha**2) / sigma**2)
    )
    return a1 + a2


def st_logli(dp, y):
    """the negative log-likelihood function of skew-t distribution

    Parameters
    ----------
    dp : array | list
        the direct parameters of the skew-t distribution in order: alpha, degrees of freedom, location and scale.
    y : array
        the data

    Returns
    -------
    float
        the negative log-likelihood function
    """
    a, df, loc, scale = dp[0], dp[1], dp[2], dp[3]
    if scale <= 0 or df <= 0:
        return np.nan
    else:
        logL = np.sum(skewt.logpdf(y, a=a, df=df, loc=loc, scale=scale))
        logL = np.where(np.isinf(logL), -1e6, logL)
    return -2 * logL


def skewt_fit(data, init_params=np.array([1, 1, 1, 1])):
    """This function fit the parameters of skew-t distribution by using the algorithm `L-BFGS-B`

    Parameters
    ----------
    data : 1D array
        the sample data
    init_params : list, optional
        the initial values of direct parameters, by default np.array([1, 1, 1, 1])

    Returns
    -------
    array : np.array
        the estimation of direct parameters: location, scale, shape, and degree of freedom.

    Raises
    ------
    ValueError
        the message from minimizer
    """
    bounds = Bounds([-np.inf, 1e-4, -np.inf, 1e-4], [np.inf, np.inf, np.inf, np.inf])
    res = minimize(
        fun=st_logli, x0=init_params, args=(data,), bounds=bounds, method="L-BFGS-B"
    )
    if res.success:
        fitted_params = res.x
        return fitted_params
    else:
        return np.repeat(np.nan, 4)


def metropolis_hastings(init_params, data, size=10000, burn_in=0.4, stepsize=0.5, cv=1):
    """This script carries out the metropolis hastings algorithm to sample the parameters of the skew-t distribution.

    Parameters
    ----------
    init_params : float
        the initial parameters
    data : float
        the sample data
    size : int, optional
        the smaple size of MH algorithm, by default 10000
    burn_in : float, optional
        the burn-in rate, by default 0.4
    stepsize : float, optional
        the step size for proposal, by default 0.5
    cv : int, optional
        the value in Theorem 1 of the paper, by default 1

    Returns
    -------
    float
        the samples of the parameters
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


def generate_single_subject(
    n1, n2, alpha, df, loc, scale, sig_level, alternative, threshold=20
):
    """This function generates a single-subject observations based on the given parameters.

    Parameters
    ----------
    n1 : int
        the number of positive observations
    n2 : int
        the number of negative observations
    alpha : float
        the skewness of the skew-t distribution
    df : float
        the degrees of freedom
    loc : float
        the location parameter
    scale : float
        the scale parameter
    sig_level : float
        the significance level
    alternative : string
        the direction of alternative hypothesis: "two_sided", "less", or "greater"
    threshold : int, optional
        the threshold for data generation, by default 20

    Returns
    -------
    float
        the single-subject observations
    """
    if alternative == "two_sided":
        q = [sig_level / 2, sig_level, 1 - sig_level, 1 - sig_level / 2]
        quantile = skewt.ppf(q, a=alpha, df=df, loc=loc, scale=scale)
        x_single_s1_1 = np.random.uniform(
            -threshold + quantile[0], quantile[0], int(n1 * 0.5)
        )
        x_single_s1_2 = np.random.uniform(
            quantile[3], threshold + quantile[3], n1 - int(n1 * 0.5)
        )
        x_single_s2_1 = np.random.uniform(quantile[0], quantile[1], int(n2 * 0.5))
        x_single_s2_2 = np.random.uniform(quantile[2], quantile[3], n2 - int(n2 * 0.5))
        x_single = np.concatenate(
            [x_single_s1_1, x_single_s1_2, x_single_s2_1, x_single_s2_2]
        )
    elif alternative == "less":
        q = [sig_level, 2 * sig_level]
        quantile = skewt.ppf(q, a=alpha, df=df, loc=loc, scale=scale)
        x_single_s1 = np.random.uniform(-threshold + quantile[0], quantile[0], n1)
        x_single_s2 = np.random.uniform(quantile[0], quantile[1], n2)
        x_single = np.concatenate([x_single_s1, x_single_s2])
    elif alternative == "greater":
        q = [1 - 2 * sig_level, 1 - sig_level]
        quantile = skewt.ppf(q, a=alpha, df=df, loc=loc, scale=scale)
        x_single_s1 = np.random.uniform(quantile[1], quantile[1] + threshold, n1)
        x_single_s2 = np.random.uniform(quantile[0], quantile[1], n2)
        x_single = np.concatenate([x_single_s1, x_single_s2])

    underlying = np.concatenate([np.repeat("P", n1), np.repeat("N", n2)])
    return x_single, underlying


def compute_fdr_acc(
    data,
    init_params,
    x_single,
    stepsize,
    underlying,
    sig_level,
    alternative,
    each=100,
    size=10000,
    burn_in=0.5,
    cv=1,
):
    """This function computes the false discovery rate and accuracy of the skew-t distribution function.

    Parameters
    ----------
    data : float
        the data
    init_params : float array
        the initial parameters of the skew-t distribution
    x_single : float
        the single-subject observations
    stepsize : float
        the step size of MH algorithm
    underlying : logical
        the underlying truth of the single-subject observations
    sig_level : float
        the significance level
    alternative : string
        the direction of alternative hypothesis: "two_sided", "less", or "greater"
    each : int, optional
        the $s$ in paper, by default 100
    size : int, optional
        the sample size of MH algorithm, by default 10000
    burn_in : float, optional
        the burn in rate, by default 0.5
    cv : int, optional
        the value in Theorem 1 of the paper, by default 1

    Returns
    -------
    float
        the results of false discovery rate and accuracy of different methods
    """
    if alternative == "two_sided":
        quan_int = [sig_level / 2, 1 - sig_level / 2]
    elif alternative == "less":
        quan_int = sig_level
    elif alternative == "greater":
        quan_int = 1 - sig_level
    sample_param = metropolis_hastings(
        init_params, data, size=size, burn_in=burn_in, stepsize=stepsize, cv=cv
    )
    accept_rate = np.mean(np.diff(sample_param, axis=0) != 0)
    unqiue_rows, counts = np.unique(sample_param, axis=0, return_counts=True)
    sample_b = np.array([])
    for row, count in zip(unqiue_rows, counts):
        x_temp = skewt.rvs(row[0], row[1], row[2], row[3], size=count * each)
        sample_b = np.append(sample_b, x_temp)
    credible_int = np.quantile(sample_b, quan_int)

    r_mh = contingency_table(x_single, credible_int, underlying, alternative)
    # NP analysis
    conf_int = np.quantile(data, quan_int)
    r_np = contingency_table(x_single, conf_int, underlying, alternative)
    #  for the MLE
    para_mle = skewt_fit(data, init_params)
    conf_int_mle = skewt.ppf(
        quan_int, para_mle[0], para_mle[1], para_mle[2], para_mle[3]
    )
    r_mle = contingency_table(x_single, conf_int_mle, underlying, alternative)
    # for MAP
    para_map = map_posterior_global(data, cv=1, init_params=init_params)
    conf_int_map = skewt.ppf(
        quan_int, para_map[0], para_map[1], para_map[2], para_map[3]
    )
    r_map = contingency_table(x_single, conf_int_map, underlying, alternative)

    return np.stack([r_mh, r_map, r_np, r_mle], axis=0), accept_rate


def contingency_table(x_single, interval, underlying, alternative):
    # underlying: 0 represents H0, i.e negative; 1 represents H1, i.e. positive
    if alternative == "two_sided":
        pred_positive = np.logical_or(x_single < interval[0], x_single > interval[1])
        pred_negative = np.logical_and(x_single >= interval[0], x_single <= interval[1])
    elif alternative == "less":
        pred_positive = x_single < interval
        pred_negative = x_single >= interval
    elif alternative == "greater":
        pred_positive = x_single > interval
        pred_negative = x_single <= interval
    n_tp = np.sum(np.logical_and(pred_positive, underlying == "P"))
    n_fp = np.sum(np.logical_and(pred_positive, underlying == "N"))
    n_fn = np.sum(np.logical_and(pred_negative, underlying == "P"))
    n_tn = np.sum(np.logical_and(pred_negative, underlying == "N"))
    return np.array([n_tp, n_fp, n_fn, n_tn])


def map_posterior_global(data, cv, init_params=np.array([1, 1, 1, 1])):
    """This function computes the MAP estimate of the skew-t distribution using the global optimization algorithm."""
    bounds = Bounds([-np.inf, 1e-4, -np.inf, 1e-4], [np.inf, 1e8, np.inf, 1e8])
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "args": (data, cv),
        "tol": 1e-10,
    }
    res = basinhopping(
        neg_log_posterior, x0=init_params, minimizer_kwargs=minimizer_kwargs, niter=100
    )

    if res.lowest_optimization_result.success:
        fitted_params = res.x
        return fitted_params
    else:
        return np.repeat(np.nan, 4)
