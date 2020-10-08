#!/usr/bin/python
# -*- coding: utf-8 -*-

from abra.config import DEFAULT_ALPHA, MIN_OBS_FOR_Z
from abra.stats import Samples, MeanComparison
from abra.inference.frequentist.results import FrequentistTestResults
from abra.inference import FrequentistProcedure
from numpy import ndarray


class MeansDelta(FrequentistProcedure):
    """
    Runs frequentist inference procedure to test for the difference in two sample
    means. Assumes normality for large sample sizes, or t-distribution for small
    sample sizes.

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
        - 'unequal': use Smith-Satterthwait dof when calculating t-stat
    """
    def __init__(self, var_assumptions='unequal', *args, **kwargs):
        super(MeansDelta, self).__init__(*args, **kwargs)
        self.var_assumptions = var_assumptions

    def run(self, control_samples, variation_samples,
            alpha=DEFAULT_ALPHA, inference_kwargs=None):
        """
        Run the inference procedure over the samples with a selected alpha
        value.

        alpha : float in [0, 1]
            the assumed Type I error rate
        """
        if isinstance(control_samples, (list, ndarray)):
            control_samples = Samples(control_samples)

        if isinstance(variation_samples, (list, ndarray)):
            variation_samples = Samples(variation_samples)

        self.alpha = alpha

        nobs = min(control_samples.nobs, variation_samples.nobs)
        test_statistic = 'z' if nobs > MIN_OBS_FOR_Z else 't'
        self.comparison = MeanComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            test_statistic=test_statistic,
            hypothesis=self.hypothesis
        )

    @property
    def stats(self):
        f_stats = getattr(self.comparison, "{}test_ind".format(self.test_statistic))
        return f_stats(alternative=self.hypothesis_sm)

    @property
    def ci(self):
        """
        Calculate confidence interval percentiles and values.
        """
        f_ci = getattr(self.comparison, "{}confint_diff".format(self.test_statistic))
        ci_vals = f_ci(self.alpha, self.hypothesis_sm, self.var_assumptions)

        return [ci_vals, self.ci_percents]

    def make_results(self):
        """
        Package up inference results
        """
        stats = self.stats
        statistic_value = stats[0]
        p_value = stats[1]

        accept_hypothesis = self.accept_hypothesis(statistic_value)
        df = stats[2] if self.test_statistic == 't' else None

        return FrequentistTestResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            delta=self.comparison.delta,
            delta_relative=self.comparison.delta_relative,
            effect_size=self.comparison.effect_size,
            alpha=self.comparison.alpha,
            power=self.comparison.power,
            confidence_interval=self.ci,
            test_statistic=self.test_statistic,
            statistic_value=statistic_value,
            p_value=p_value,
            df=df,
            hypothesis=self.hypothesis_text,
            accept_hypothesis=accept_hypothesis,
            inference_procedure=self,
            warnings=self.comparison.warnings
        )
