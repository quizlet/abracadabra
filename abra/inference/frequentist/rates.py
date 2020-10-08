#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.config import DEFAULT_ALPHA
from abra.stats import Samples, RateComparison
from abra.inference.frequentist.results import FrequentistTestResults
from abra.inference import FrequentistProcedure
import numpy as np
from scipy.stats import norm


class RatesRatio(FrequentistProcedure):
    """
    Runs frequentist inference procedure to test for the difference in two sample
    rates.

    Parameters
    ----------
    hypothesis : str
        the althernative hypothesis:
        - 'larger': one-tailed test, assume variation is larger than null
        - 'smaller': one-tailed test, assume variation is smaller than null
        - 'unequal': two-tailed test, variaion mean is different than null
    var_assumptions : str
        whether to use pooled or unequal variance assumptions
        - 'pooled': assume the same variance
        - 'unequal': use Smith-Satterthwait dof when cal1ualting t-stat
    """
    def __init__(self, *args, **kwargs):
        super(RatesRatio, self).__init__(*args, **kwargs)

    def run(self, control_samples, variation_samples,
            alpha=DEFAULT_ALPHA, inference_kwargs=None):
        """
        Run the inference procedure over the samples with a selected alpha
        value.

        alpha : float in [0, 1]
            the assumed Type I error rate
        """
        if isinstance(control_samples, (list, np.ndarray)):
            control_samples = Samples(control_samples)

        if isinstance(variation_samples, (list, np.ndarray)):
            variation_samples = Samples(variation_samples)

        self.alpha = alpha
        self.comparison = RateComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            hypothesis=self.hypothesis
        )

    @property
    def stats(self):
        if not hasattr(self, '_stats'):
            self._stats = self.comparison.rates_test()
        return self._stats

    @property
    def ci(self):
        """
        Calculate confidence interval for  rates ratio. Intervals outside of 1
        support the alternative hypothesis.

        Calculation follows Li, Tang, & Wong, 2008, using the MOVER-R method,
        and Rao Score intervals (Altman et al, 2000) for individual rate interval
        estimates (aka "FRSI")

        Returns
        -------
        CIs : list
            [(CI_lower, CI_upper), (CI_lower_percentile, CI_upper_percentile)]

        References
        ----------
        Li H.Q, Tang ML, Wong WK (2008) Confidence intervals for ratio of two
            Poisson rates using the method of variance estimates recovery.
            Biometrical Journal 50 (2008)
        Altman D., Machin D., Bryant TN. et al. (2000) "Statistics with confidence"
            (2nd). BMJ Books: Bristol.
        """

        def rao_score_interval(X, z, t):
            # individual rate interval method 2 (Altman et al., 2000)
            a = X + .5 * z**2.
            b = z * np.sqrt(X + .25 * z**2.)
            return (a - b) / t, (a + b) / t

        if self.hypothesis == 'larger':
            z = norm.ppf(1 - self.alpha)
        elif self.hypothesis == 'smaller':
            z = norm.ppf(self.alpha)
        elif self.hypothesis == 'unequal':
            z = np.abs(norm.ppf(1 - self.alpha / 2.))

        control = self.comparison.d2
        variation = self.comparison.d1

        X1, t1 = control.data.sum(), control.nobs
        X2, t2 = variation.data.sum(), variation.nobs

        lam_1 = X1 / t1
        lam_2 = X2 / t2

        lam_2_lam_1 = lam_2 * lam_1

        l2, u2 = rao_score_interval(X1, z, t1)
        l1, u1 = rao_score_interval(X2, z, t2)

        # Gu et al, 2008; Eq 3
        L = (lam_2_lam_1 - np.sqrt(lam_2_lam_1 ** 2 - l1 * (2 * lam_2 - l1) * (u2 * (2 * lam_1 - u2)))) /  (u2 * (2 * lam_1 - u2))

        # Gu et al, 2008; Eq 4
        U = (lam_2_lam_1 + np.sqrt(lam_2_lam_1 ** 2 - u1 * (2 * lam_2 - u1) * (l2 * (2 * lam_1 - l2)))) / (l2 * (2 * lam_1 - l2))

        return [(L, U), self.ci_percents]

    def make_results(self):
        """
        Package up inference results
        """
        statistic_value, p_value = self.stats
        accept_hypothesis = self.accept_hypothesis(statistic_value)

        return FrequentistTestResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            delta=self.comparison.delta,
            delta_relative=self.comparison.delta_relative,
            effect_size=self.comparison.effect_size,
            alpha=self.comparison.alpha,
            confidence_interval=self.ci,
            test_statistic=self.test_statistic,
            statistic_value=statistic_value,
            p_value=p_value,
            hypothesis=self.hypothesis_text,
            accept_hypothesis=accept_hypothesis,
            inference_procedure=self,
            power=self.comparison.power
        )
