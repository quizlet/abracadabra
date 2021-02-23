#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.config import DEFAULT_ALPHA, logger
from abra.mixin import InitRepr
from statsmodels.stats.api import DescrStatsW, CompareMeans
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest, binom_test
from scipy.stats import norm
from scipy import optimize
from pandas import DataFrame
import numpy as np


CORRECTIONS = {'b': 'bonferroni', 's': 'sidak', 'bh': 'fdr_bh'}


def bonferroni(alpha_orig, p_values):
    """
    Bonferrnoi correction.

    en.wikipedia.org/wiki/Bonferroni_correction

    Parameters
    ----------
    alpha_orig : float
        alpha value before correction
    p_values: list[float]
        p values resulting from all the tests

    Returns
    -------
    alpha_corrected: float
        new critical value (i.e. the corrected alpha)
    """
    return alpha_orig / len(p_values)


def sidak(alpha_orig, p_values):
    """
    Sidak correction.

    en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction

    Parameters
    ----------
    alpha_orig : float
        alpha value before correction
    p_values: list[float]
        p values resulting from all the tests

    Returns
    -------
    alpha_corrected: float
        new critical value (i.e. the corrected alpha)
    """
    return 1. - (1. - alpha_orig) ** (1. / len(p_values))


def fdr_bh(fdr, p_values):
    """
    Benjamini-Hochberg false-discovery rate adjustment procedure.

    pdfs.semanticscholar.org/af6e/9cd1652b40e219b45402313ec6f4b5b3d96b.pdf

    Parameters
    ----------
    fdr : float
        False Discovery Rate (q*), proportion of significant results that are
        actually false positives
    p_values: list[float]
        p values resulting from all the tests

    Returns
    -------
    alpha_corrected: float
        new critical value (i.e. the corrected alpha)
    """
    n_tests = len(p_values)

    def p_i(i):
        return i * fdr / n_tests

    p_sorted = np.sort(np.asarray(p_values))

    significant_idx = [i for i, val in enumerate(p_sorted, 1) if val <= p_i(i)]
    rank = np.max(significant_idx) if significant_idx else 1
    return p_i(rank)


def estimate_experiment_sample_sizes(
    delta,
    statistic='z',
    alpha=.05,
    power=.8,
    *args, **kwargs
):
    """
    Calculate the sample size required for each treatement in order to observe a
    difference of `delta` between control and variation groups, for a given setting
    of `alpha`, `power`.

    Parameters
    ----------
    delta : float
        The absolute difference in means between control and variation groups
    statistic : string
        Either:
            - 'z' or 't' if interpreting effect size as scaled difference of means
            - 'rates_ratio' if interpreeting effect size as the ratio of means
    alpha : float [0, 1)
            The assumed Type I error of the test
    power : float [0, 1)
        The desired statistical power of the test
    *args, **kwargs
        Model-specific arguments

    Returns
    -------
    sample_sizes : list[int]
        The estiamated sample sizes for the control and variation treatments

    Example 1: Continuous Variables
    -------------------------------
    # Estimate the sample size required to observe significant difference between
    # two binomial distributions that differ by .01 in mean probability with
    # Type I error = 0.05 (default) and Power = 0.8 (default)

    prob_control = .49
    std_control = (prob_control * (1 - prob_control))**.5  # Binomial std
    prob_variation = std_variation = .50
    delta = prob_variation - prob_control

    print(
        estimate_experiment_sample_sizes(
            delta=delta,
            statistic='z',
            std_control=std_control,
            std_variation=std_variation
        )
    )
    # [39236, 39236]

    Example 2 - Count Variables
    ---------------------------
    # Replicate Example 1 from Gu et al, 2008

    R = 4  # ratio under alternative hypothesis
    control_rate = .0005
    variation_rate = R * control_rate
    delta = variation_rate - control_rate

    print(
        estimate_experiment_sample_sizes(
            delta,
            statistic='rates_ratio',
            control_rate=control_rate,
            alpha=.05,
            power=.9,
            control_exposure_time=2.,
            sample_size_ratio=.5
        )
    )
    # [8590, 4295]
    """
    if statistic in ('t', 'z'):
        # std_control and/or std_variation are in *args, or **kwargs
        return cohens_d_sample_size(delta, alpha, power, statistic, *args, **kwargs)
    elif statistic == 'rates_ratio':
        return ratio_sample_size(alpha, power, delta, *args, **kwargs)
    else:
        raise ValueError("Unknown statistic")


def cohens_d(delta, std_control, std_variation=None):
    std_variation = std_variation if std_variation else std_control
    std_pooled = np.sqrt((std_control ** 2 + std_variation ** 2) / 2.)
    return delta / std_pooled


def cohens_d_sample_size(
    delta,
    alpha,
    power,
    statistic,
    std_control,
    std_variation=None,
    sample_size_ratio=1.
):
    """
    Calculate sample size required to observe a significantly reliable difference
    between groups a and b. Assumes Cohen's d definition of effect size and an
    enrollment ratio of 1.0 between groups a and b by default.

    Parameters
    ----------
    std_control : float
        An estiamte of the expected sample standard deviation of control
        group
    nobs_control : int
        The number of control observations.
    std_variation : float
        An estimate of the expected sample standard deviation of variation
        group. If not provided, we assume homogenous variances for the
        two groups.

    Returns
    -------
    sample_sizes : list[int]
        The estiamated sample sizes for the control and variation treatments

    Example
    -------
    # Get estimate of sample size required to observe a significant difference between
    # two binomial distributions that differ by .01 in mean probability

    prob_control = .49
    std_control = (prob_control * (1 - prob_control))**.5  # Binomial std
    prob_variation = std_variation = .50
    delta = prob_variation - prob_control

    print(
        cohens_d_sample_size(
            delta=delta,
            alpha=.05,
            power=.8,
            statistic='z',
            std_control=std_control,
            std_variation=std_variation
        )
    )
    # [39236, 39236]

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences
        (2nd ed.). Hillsdale, NJ: Lawrence Earlbaum Associates.
    """
    SUPPORTED_STATISTICS = ('t', 'z')
    effect_size = cohens_d(delta, std_control, std_variation)

    if statistic in SUPPORTED_STATISTICS:
        power_func = "{}t_ind_solve_power".format(statistic)
        N1 = int(
            eval(power_func)(
                effect_size,
                alpha=alpha,
                power=power,
                ratio=sample_size_ratio
            )
        )
        N2 = int(N1 * sample_size_ratio)
        return [N1, N2]
    else:
        raise ValueError("Unknown statistic, must be either {!r}".format(SUPPORTED_STATISTICS))


def ratio_sample_size(
    alpha,
    power,
    delta,
    control_rate,
    control_exposure_time=1.,
    null_ratio=1.,
    sample_size_ratio=1.,
    exposure_time_ratio=1.
):
    """
    Calculate sample size required to observe a significantly reliable ratio of
    rates between variation and control groups. Follows power calculation outlined
    in Gu et al, 2008.

    Parameters
    ----------
    control_rate : float
        The poisson rate of the control group
    control_exposure_time : float
        The number of time units of the control exposure. Default is 1.0
    null_ratio : float
        The ratio of variation to control rates under the null hypothesis.
        Default is 1.
    sample_size_ratio : float
        The ratio of sample sizes of the variation to the control groups. Default is
        1, thus assuming equal sample sizes.
    exposure_time_ratio : float
        The ratio of the variation exposure time to the control. Default is 1.0,
        thus assuming equal exposure times

    Returns
    -------
    N1, N2 : tuple
        Sample sizes for each group

    Example
    -------
    # Replicate Example 1 from Gu et al, 2008

    R = 4  # ratio under alternative hypothesis
    control_rate = .0005
    variation_rate = R * control_rate
    delta = variation_rate - control_rate

    print(
        ratio_sample_size(
            alpha=.05,
            power=.9,
            delta=delta,
            control_rate=control_rate,
            control_exposure_time=2.,
            sample_size_ratio=.5
        )
    )
    # returns [8590, 4295], which have been validated to be more accurate than
    # the result reported in Gu et al, due to rounding precision. For details
    # see "Example 2 – Validation using Gu (2008)" section of
    # http://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tests_for_the_Ratio_of_Two_Poisson_Rates.pdf

    References
    ----------
    Gu, K., Ng, H.K.T., Tang, M.L., and Schucany, W. 2008. 'Testing the Ratio of
        Two Poisson Rates.' Biometrical Journal, 50, 2, 283-298.
    Huffman, M. 1984. 'An Improved Approximate Two-Sample Poisson Test.'
        Applied Statistics, 33, 2, 224-226.
    """

    # convert absolute difference to ratio
    alternative_ratio = float(control_rate + delta) / control_rate
    variation_exposure_time = exposure_time_ratio * control_exposure_time

    z_alpha = norm.ppf(1 - alpha)
    z_power = norm.ppf(power)

    def objective(x):
        ratio_proposed = (x[1] * variation_exposure_time) / (x[0] * control_exposure_time)
        loss = np.abs(null_ratio - (alternative_ratio / ratio_proposed))
        return loss

    def con1(x):
        """General sample size ratio constraint"""
        return (float(x[1]) / x[0]) - sample_size_ratio

    def con2(x):
        """Control sample size constraint, outlined in Gu et al, 2008, Equation 10"""

        N1, N2 = x
        d = (control_exposure_time * N1) / (variation_exposure_time * N2)
        A = 2 * (1. - np.sqrt(null_ratio / alternative_ratio))
        C = np.sqrt((null_ratio + d) / alternative_ratio)
        D = np.sqrt((alternative_ratio + d) / alternative_ratio)
        return x[0] - (((z_alpha * C + z_power * D) / A) ** 2. - (3. / 8)) / (control_exposure_time * control_rate)

    constraint1 = {'type': 'eq', 'fun': con1}
    constraint2 = {'type': 'eq', 'fun': con2}
    constraints = [constraint1, constraint2]

    results = optimize.minimize(
        objective,
        (10, 10),
        bounds=((1, None), (1, None)),
        constraints=constraints,
        method='SLSQP',
        tol=1e-10
    )

    return [int(np.ceil(n)) for n in results.x]


class MultipleComparisonCorrection(InitRepr):
    """
    Perform multiple comparison adjustment of alpha based on a sequence of
    p_values that result from two or more hypothesis tests inference procedures.

    param p_values : list[float]
        A list of p_values resulting from two or more hypothesis tests.
    method :  str
        One of the following correction methods:
            'bonferroni', 'b' : one-step Bonferroni correction
            'sidak', 's' : one-step Sidak correction
            'fdr_bh', 'bh; : Benjamini/Hochberg (non-negative)
    alpha : float in (0, 1)
        the desired probability of Type I error
    reject_nul: list[bool]
        For each probablity, whether or not to reject the null hypothsis given
        the updated values for alpha.
    """
    __ATTRS__ = ['ntests', 'method', 'alpha_orig', 'alpha_corrected']

    def __init__(self, p_values, method='sidak', alpha=DEFAULT_ALPHA):
        if method not in set(list(CORRECTIONS.keys()) + list(CORRECTIONS.values())):
            raise ValueError('Correction method {!r} not supported'.format(method))

        self.method = CORRECTIONS[method] if method in CORRECTIONS else method
        self.alpha_orig = alpha
        self.alpha_corrected = eval(self.method)(alpha, p_values)
        self.ntests = len(p_values)
        self.accept_hypothesis = [p < self.alpha_corrected for p in p_values]


class EmpiricalCdf(object):
    """
    Class that calculates the empirical cumulative distribution function for a
    set of samples. Performs some additional cacheing for performance.
    """
    def __init__(self, samples):
        self.samples = samples
        self._cdf = ECDF(samples)

    @property
    def samples_cdf(self):
        """
        Return the cdf evaluated at those samples used to calculate the cdf
        parameters.
        """
        if not hasattr(self, '_samples_cdf'):
            self._samples_cdf = self.evaluate(sorted(self.samples))
        return self._samples_cdf

    def __call__(self, values):
        return self.evaluate(values)

    def evaluate(self, values=None):
        """
        Evaluate the cdf for a sequence of values
        """
        if values is None:
            values = self.samples
        return self._cdf(values)


class Samples(DescrStatsW):
    """
    Class for holding samples and calculating various statistics on those
    samples.

    Parameters
    ----------
    samples: array-like
        the data set of sample values
    """
    def __init__(self, observations, name=None):

        self.name = name
        observations = self._valid_observations(observations)
        super(Samples, self).__init__(np.array(observations))

    def _valid_observations(self, observations):
        def valid(o):
            if o is None:
                return False
            if np.isnan(o):
                return False
            return True

        observations = list(filter(valid, observations))
        if self.name:
            name_string = "{!r}".format(self.name)
        else:
            name_string = ''
        if not observations:
            raise ValueError('All {} observations are nan or None'.format(name_string))
        else:
            return observations

    def __repr__(self):
        header = "Samples(name={!r})".format(self.name if self.name else None)
        return """{}
Summary:
𝛮  : {}
𝝁  : {:1.4f}
𝝈² : {:1.4f}""".format(header, self.nobs, self.mean, self.var)

    def permute(self):
        return np.random.choice(self.data, int(self.nobs))

    def sort(self):
        if not hasattr(self, '_sorted'):
            self._sorted = sorted(self.data)
        return self._sorted

    def percentiles(self, prct=[2.5, 25, 50, 75, 97.5]):
        return np.percentile(self.data, prct)

    @property
    def cdf(self):
        if not hasattr(self, '_cdf'):
            self._cdf = EmpiricalCdf(self.data)
        return self._cdf

    def prob_greater_than(self, values):
        """
        Return the probability of being larger than values under the emprical
        CDF
        """
        return 1.0 - self.cdf(np.asarray(values, dtype=float))

    def ci(self, alpha=.05, alternative='two-sided'):
        """
        Calculate the (1-alpha)-th confidence interval around the mean.
        Assumes Gaussian approximation.

        Returns
        -------
        ci : tuple (lo, hi)
            the (1-alpha) % confidence interval around the mean estimate.
        """
        return self.zconfint_mean(alpha, alternative)[:2]

    def std_err(self, alpha=.05, alternative='two-sided'):
        """
        Returns
        -------
        std_err : tuple (lo, hi)
            the standard error interval around the mean estimate.
        """
        _alpha = alpha / 2. if alternative == 'two-sided' else alpha
        z = norm.ppf(1 - _alpha)
        ci = z * (self.var / self.nobs) ** .5
        return self.mean - ci, self.mean + ci

    def hdi(self, alpha=.05):
        """
        Calcualte the highest central density interval that leaves `alpha`
        probability remaining.

        Parameters
        ----------
        alpha: float in (0, 1)
            1 - critical mass

        Returns
        -------
        hdi: tuple (boundary_lower, boundary_upper)
            The boundary of the highest density interval for the sample distribution
        """
        credible_mass = 1 - alpha
        try:
            _hdi = highest_density_interval(self.data, credible_mass)
            return (round(_hdi[0], 4), round(_hdi[1], 4))
        except Exception as e:
            logger.warn(e)
            return (None, None)

    def hist(self, ref_val=None, *hist_args, **hist_kwargs):
        """
        Render histogram of the samples. Plot a vertical reference line, if
        requested.

        """
        from matplotlib import pyplot as plt
        pl = plt.hist(self.data.astype(float), *hist_args, **hist_kwargs)
        if ref_val is not None:
            plt.axvline(ref_val, c='gray', linestyle='--', linewidth=2)
        return pl

    def plot_probability(self, *args, **kwargs):
        """
        Evaulate and display the sample probability function.
        """
        self.prob.plot(*args, **kwargs)


class MeanComparison(CompareMeans):
    """
    Class for comparing the means of two sample distributions, provides a number
    of helpful summary statistics about the comparison.

    Parameters
    ----------
    samples_a : Samples instance
        Group a samples
    samples_b : Samples instance
        Group b samples
    alpha : float in (0, 1)
        The assumed Type I error
    test_statistic: str
        The name of the test statistic used.
            't': for t-statistic (small sample size, N <= 30)
            'z': for z-statistic (large samples size, N > 30)
    hypothesis : str
        Defines the assumed alternative hypothesis. Can be :
            'larger'
            'smaller'
            'unequal' (i.e. two-tailed test)
    """
    def __init__(self, samples_a, samples_b,
                 alpha=DEFAULT_ALPHA,
                 test_statistic='t',
                 hypothesis='larger'):

        super(MeanComparison, self).__init__(samples_a, samples_b)

        self.alpha = alpha
        self.test_statistic = test_statistic
        self.hypothesis = hypothesis
        self.warnings = []

    @property
    def pooled_variance(self):
        return ((self.d2.nobs - 1) * self.d2.var + (self.d1.nobs - 1) * self.d1.var) / (self.d2.nobs + self.d1.nobs - 2)

    @property
    def delta(self):
        return self.d1.mean - self.d2.mean

    @property
    def delta_relative(self):
        return (self.d1.mean - self.d2.mean) / np.abs(self.d2.mean)

    @property
    def effect_size(self):
        return self.delta / np.sqrt(self.pooled_variance)

    @property
    def test_direction(self):
        return self.hypothesis if self.hypothesis != 'unequal' else 'two-sided'

    @property
    def power(self):
        """
        Statistical power (i.e. 𝜷 of the comparison)
        """
        ratio = self.d1.nobs / self.d2.nobs

        f_stat = "{}t_ind_solve_power".format(self.test_statistic)

        return eval(f_stat)(
            effect_size=self.effect_size,
            nobs1=self.d2.nobs,
            alpha=self.alpha,
            ratio=ratio,
            alternative=self.test_direction
        )


class ProportionComparison(MeanComparison):
    """
    Class for comparing the proportions of two sample distributions, provides a number
    of helpful summary statistics about the comparison. In order to use the
    z-distribution, we assume normality or proportions and thus, by proxy, adequate
    sample sizes (i.e. > 30).

    Parameters
    ----------
    samples_a : Samples instance
        Group a samples
    samples_b : Samples instance
        Group b samples
    alpha : float in (0, 1)
        The assumed Type I error
    hypothesis : str
        Defines the assumed alternative hypothesis. Can be :
            'larger'
            'smaller'
            'unequal' (i.e. two-tailed test)
    """
    def __init__(self, variance_assumption='pooled', *args, **kwargs):

        super(ProportionComparison, self).__init__(test_statistic='z', *args, **kwargs)
        nobs = min(self.d1.nobs, self.d2.nobs)

        # to use Normal approx, must have large N
        if nobs < 30:
            warning = 'Normality assumption violated, at least 30 observations required. Smallest sample size is {}'.format(nobs)
            logger.warn(warning)
            self.warnings.append(warning)

        self.variance_assumption = variance_assumption

    @property
    def pooled_variance(self):
        if self.variance_assumption == 'pooled':
            p1 = self.d1.mean
            p2 = self.d2.mean
            var1 = p1 * (1 - p1)
            var2 = p2 * (1 - p2)
            return ((self.d1.nobs - 1) * var1 + (self.d2.nobs - 1) * var2) / (self.d1.nobs + self.d2.nobs - 2)
        else:  # global variance
            p = np.mean(np.r_[self.d1.data, self.d2.data])
            return p * (1 - p)

    def ztest(self):
        prop_var = self.pooled_variance
        n_1 = self.d1.nobs
        s_1 = sum(self.d1.data)
        n_2 = self.d2.nobs
        s_2 = sum(self.d2.data)
        return proportions_ztest(
            [s_1, s_2], [n_1, n_2],
            alternative=self.test_direction,
            prop_var=prop_var
        )


class RateComparison(MeanComparison):
    """
    Class for comparing the rates of two sample distributions, provides a number
    of helpful summary statistics about the comparison. Uses the exact conditional
    test based on binomial distribution, as described in Gu et al (2008)

    Parameters
    ----------
    samples_a : Samples instance
        Group a samples
    samples_b : Samples instance
        Group b samples
    alpha : float in (0, 1)
        The assumed Type I error
    hypothesis : str
        Defines the assumed alternative hypothesis. Can be :
            'larger'
            'smaller'
            'unequal' (i.e. two-tailed test)

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008
    """
    def __init__(self, null_ratio=1., *args, **kwargs):

        super(RateComparison, self).__init__(test_statistic='W', *args, **kwargs)
        self.null_ratio = null_ratio

    @property
    def rates_ratio(self):
        """
        Return the comparison ratio of the null rates ratio and the observed
        rates ratio.
        """
        actual_ratio = float(self.d1.sum * self.d1.nobs) / float(self.d2.sum * self.d2.nobs)
        return self.null_ratio / actual_ratio

    @property
    def delta(self):
        """
        Delta is the ratio of the variation to the control rates
        """
        return self.d1.mean / self.d2.mean

    @property
    def delta_relative(self):
        return self.delta

    def rates_test(self):
        """
        Run the rates comparison hyptothesis test. Uses the W5 statistic defined
        in Gu et al., 2008

        Returns
        -------
        W : float
            The W5 statistic from Gu et al., 2008
        p_value : float
            The p-value associated with W
        """
        X1, X2 = self.d2.sum, self.d1.sum
        t1, t2 = self.d2.nobs, self.d1.nobs
        d = float(t1) / t2
        W = 2 * (np.sqrt(X2 + (3. / 8)) - np.sqrt((self.null_ratio / d) * (X1 + (3. / 8)))) / np.sqrt(1 + (self.null_ratio / d))

        if self.hypothesis == 'larger':
            p_val = 1 - norm.cdf(W)
        elif self.hypothesis == 'smaller':
            p_val = norm.cdf(W)
        elif self.hypothesis == 'unequal':
            p_val = 1 - norm.cdf(abs(W))

        return W, p_val

    @property
    def effect_size(self):
        """
        Effect size ranges from 0-1
        """
        return 1 - self.rates_ratio

    @property
    def power(self):
        """
        Return the statistical power of the current test. Follows the calculation
        from W statistic 5 in Gu et al., 2008
        """
        N2, t2 = self.d1.sum, self.d1.nobs
        N1, t1 = self.d2.sum, self.d2.nobs

        lambda_2, lambda_1 = np.abs(self.d1.mean), np.abs(self.d2.mean)
        alternative_ratio = np.abs(lambda_2 / lambda_1)
        z = norm.ppf(1 - self.alpha)

        d = float(t1 * N1) / (t2 * N2)

        A = np.abs(2. * (1. - np.sqrt(self.null_ratio / alternative_ratio)))
        B = np.sqrt(lambda_1 * t1 * N1 + (3. / 8))
        C = np.sqrt((self.null_ratio + d) / alternative_ratio)
        D = np.sqrt((alternative_ratio + d) / alternative_ratio)
        W = (A * B - z * C) / D

        return round(norm.cdf(W), 4)


def highest_density_interval(samples, mass=.95):
    """
    Determine the bounds of the interval of width `mass` with the highest density
    under the distribution of samples.

    Parameters
    ----------
    samples: list
        The samples to compute the interval over
    mass: float (0, 1)
        The credible mass under the empricial distribution

    Returns
    -------
    hdi: tuple(float)
        The lower and upper bounds of the highest density interval
    """
    _samples = np.asarray(sorted(samples))
    n = len(_samples)

    interval_idx_inc = int(np.floor(mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = _samples[interval_idx_inc:] - _samples[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = _samples[min_idx]
    hdi_max = _samples[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


class BootstrapStatisticComparison(MeanComparison):
    """
    Class for comparing a bootstrapped test statistic for two samples. Provides
    a number of helpful summary statistics about the comparison.

    Parameters
    ----------
    samples_a : Samples instance
        Group a samples
    samples_b : Samples instance
        Group b samples
    alpha : float in (0, 1)
        The assumed Type I error
    hypothesis : str
        Defines the assumed alternative hypothesis. Can be :
            'larger'
            'smaller'
            'unequal' (i.e. two-tailed test)
    n_bootstraps : int
        The number of bootstrap samples to draw use for estimates.
    statistic_function : function
        Function that returns a scalar test statistic when provided a sequence
        of samples.

    References
    ----------
    Efron, B. (1981). "Nonparametric estimates of standard error: The jackknife, the bootstrap and other methods". Biometrika. 68 (3): 589–599
    """
    def __init__(self, n_bootstraps=1000, statistic_function=None, *args, **kwargs):
        statistic_function = statistic_function if statistic_function else np.mean
        statistic_name = statistic_function.__name__
        super(BootstrapStatisticComparison, self).__init__(
            test_statistic=f"{statistic_name}", *args, **kwargs
        )
        self.statistic_function = statistic_function
        self.n_bootstraps = n_bootstraps


    def bootstrap_test(self):
        """
        Run the sample comparison hyptothesis test. Uses the bootstrapped sample statistics

        Returns
        -------
        delta: float
            The observed difference in test statistic
        p_value : float
            The p-value associated with delta
        """
        all_samples = np.concatenate([self.d1.data, self.d2.data]).astype(float)

        d1_samples = np.random.choice(all_samples, (int(self.d1.nobs), self.n_bootstraps), replace=True)
        d1_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=d1_samples)

        d2_samples = np.random.choice(all_samples, (int(self.d2.nobs), self.n_bootstraps), replace=True)
        d2_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=d2_samples)

        control_bs_samples = np.random.choice(self.d2.data, (int(self.d2.nobs), self.n_bootstraps), replace=True)
        control_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=control_bs_samples)
        self.control_bootstrap = Samples(control_statistics, name='control')

        variation_bs_samples = np.random.choice(self.d1.data, (int(self.d1.nobs), self.n_bootstraps), replace=True)
        variation_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=variation_bs_samples)
        self.variation_bootstrap = Samples(variation_statistics, name='variation')

        # The null sampling distribution of test_statistic deltas
        self.null_dist = Samples(d2_statistics - d1_statistics, name=f'{self.test_statistic}-null')

        if self.hypothesis == 'larger':
            p_val = 1 - self.null_dist.cdf(self.delta)
        elif self.hypothesis == 'smaller':
            p_val = self.null_dist.cdf(self.delta)
        elif self.hypothesis == 'unequal':
            p_val = 1 - self.null_dist.cdf(abs(self.delta))

        return self.delta, p_val


    def confidence_interval(self, alpha=.05):
        """
        Calculate the (1-alpha)-th confidence interval around the statistic delta.
        Uses bootstrapped approximation the statistic sampling distribution.

        Returns
        -------
        ci : tuple (lo, hi)
            the (1-alpha) % confidence interval around the statistic estimate.
        """
        return  self.deltas_dist.percentiles([100 * alpha, 100 * (1-alpha)])

    @property
    def deltas_dist(self):
        if not hasattr(self, '_deltas_dist'):
            d1_samples = np.random.choice(self.d1.data, (int(self.d1.nobs), self.n_bootstraps), replace=True)
            d1_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=d1_samples)

            d2_samples = np.random.choice(self.d2.data, (int(self.d2.nobs), self.n_bootstraps), replace=True)
            d2_statistics = np.apply_along_axis(self.statistic_function, axis=0, arr=d2_samples)

            self._deltas_dist = Samples(d1_statistics - d2_statistics, name=f'{self.test_statistic}-deltas')

        return self._deltas_dist

    @property
    def delta(self):
        """
        Delta is difference in test statistics
        """
        return self.deltas_dist.mean

    @property
    def delta_relative(self):
        return self.delta / np.abs(self.statistic_function(self.d2.data))

    @property
    def power(self):
        """
        Return the statistical power of the current test. Uses
        """
        critical_value = self.null_dist.percentiles(100 * (1 - self.alpha))
        return self.deltas_dist.prob_greater_than(critical_value)


def highest_density_interval(samples, mass=.95):
    """
    Determine the bounds of the interval of width `mass` with the highest density
    under the distribution of samples.

    Parameters
    ----------
    samples: list
        The samples to compute the interval over
    mass: float (0, 1)
        The credible mass under the empricial distribution

    Returns
    -------
    hdi: tuple(float)
        The lower and upper bounds of the highest density interval
    """
    _samples = np.asarray(sorted(samples))
    n = len(_samples)

    interval_idx_inc = int(np.floor(mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = _samples[interval_idx_inc:] - _samples[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = _samples[min_idx]
    hdi_max = _samples[min_idx + interval_idx_inc]
    return hdi_min, hdi_max
