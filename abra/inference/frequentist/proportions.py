#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.config import DEFAULT_ALPHA
from abra.stats import Samples, ProportionComparison
from .results import FrequentistTestResults
from abra.inference import FrequentistProcedure


class ProportionsDelta(FrequentistProcedure):
    """
    Runs frequentist inference procedure to test for the difference in two sample
    proportions. Assumes normality and thus, by proxy, adequate sample sizes
    (i.e. > 30).

    Parameters
    ----------
    hypothesis : str
        the althernative hypothesis:
        - 'larger': one-tailed test, assume variation is larger than null
        - 'smaller': one-tailed test, assume variation is smaller than null
        - 'unequal': two-tailed test, variaion mean is different than null
    var_assumptions : False or float in (0, 1)
        whether to calculate variance based on sample or use control or
        another variance
    """
    def __init__(self, var_assumptions='pooled', *args, **kwargs):

        super(ProportionsDelta, self).__init__(*args, **kwargs)
        self.var_assumptions = var_assumptions

    def run(self, control_samples, variation_samples,
            alpha=DEFAULT_ALPHA, inference_kwargs=None):
        """
        Run the inference procedure over the samples with a selected alpha
        value.

        alpha : float in [0, 1]
            the assumed Type I error rate
        """

        if not isinstance(control_samples, Samples):
            control_samples = Samples(control_samples)

        if not isinstance(variation_samples, Samples):
            variation_samples = Samples(variation_samples)

        self.alpha = alpha
        self.comparison = ProportionComparison(samples_a=variation_samples,samples_b=control_samples,alpha=self.alpha,hypothesis=self.hypothesis)

    @property
    def stats(self):
        # if not hasattr(self, '_stats'):
        self._stats = self.comparison.ztest()
        return self._stats

    @property
    def ci(self):
        """
        Calculate confidence interval percentiles and values.
        """
        var_assumptions = self.var_assumptions if self.var_assumptions == "pooled" else "unequal"
        ci_vals = self.comparison.zconfint_diff(self.alpha, self.hypothesis_sm, var_assumptions)

        return [ci_vals, self.ci_percents]

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
            power=self.comparison.power,
            confidence_interval=self.ci,
            test_statistic=self.test_statistic,
            statistic_value=statistic_value,
            p_value=p_value,
            df=None,
            hypothesis=self.hypothesis_text,
            accept_hypothesis=accept_hypothesis,
            inference_procedure=self,
            warnings=self.comparison.warnings
        )
