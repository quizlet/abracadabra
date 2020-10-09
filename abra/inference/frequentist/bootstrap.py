#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from abra.config import DEFAULT_ALPHA
from abra.stats import Samples, BootstrapStatisticComparison
from abra.inference.frequentist.results import FrequentistTestResults
from abra.inference import FrequentistProcedure


class BootstrapDelta(FrequentistProcedure):
    """
    Runs frequentist inference procedure to test for the difference in a bootstrapped
    test statistic estimate

    Parameters
    ----------
    hypothesis : str
        the althernative hypothesis:
        - 'larger': one-tailed test, assume variation is larger than null
        - 'smaller': one-tailed test, assume variation is smaller than null
        - 'unequal': two-tailed test, variaion mean is different than null
    statistic_function : callable
        Function that returns a scalar test statistic when provided a sequence
        of samples.
    """
    def __init__(self, statistic_function=None, *args, **kwargs):
        super(BootstrapDelta, self).__init__(*args, **kwargs)
        self.statistic_function = statistic_function

    def run(
        self, control_samples, variation_samples,
        alpha=DEFAULT_ALPHA, inference_kwargs=None
    ):
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
        self.comparison = BootstrapStatisticComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
            statistic_function=self.statistic_function
        )

    @property
    def stats(self):
        self._stats = self.comparison.bootstrap_test()
        return self._stats

    @property
    def ci(self):
        """
        Calculate confidence interval around deltas with percentiles and values.
        """
        ci_vals = self.comparison.confidence_interval(self.alpha)
        return [ci_vals, self.ci_percents]

    def accept_hypothesis(self, stat_value):
        """
        Use the boostrapped estimate of teh sampling distribition to test the Null
        """
        if self.hypothesis == 'larger':
            return self.alpha > self.stat_dist.prob_greater_than(stat_value)
        elif self.hypothesis == 'smaller':
            return self.alpha > 1 - self.stat_dist.prob_greater_than(stat_value)
        elif self.hypothesis == 'unequal':
            return abs(stat_value) > abs(self.stat_dist.percentiles(100 * (1 - self.alpha/2.)))
        else:
            raise ValueError('Unknown hypothesis: {!r}'.format(self.hypothesis))

    def make_results(self):
        """
        Package up inference results
        """
        statistic_value, p_value = self.stats
        self.stat_dist = self.comparison.null_dist
        accept_hypothesis = self.accept_hypothesis(statistic_value)

        aux = {
            'control': self.comparison.control_bootstrap,
            'variation': self.comparison.variation_bootstrap
        }
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
            warnings=self.comparison.warnings,
            aux=aux
        )
