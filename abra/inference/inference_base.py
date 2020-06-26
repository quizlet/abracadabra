#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm


class InferenceProcedure(object):
    """
    Base class for all inference procedures. Must implement the
    following methods:
      - `run()`
      - `make_results()`
    """
    def __init__(self, method=None, stat_dist=norm, *args, **kwargs):
        self.method = method
        self.stat_dist = stat_dist

    def run(self, control_samples, variation_samples,
            alpha=None, inference_kwargs=None):
        """
        Run inference procedure and return associated results
        """
        raise NotImplementedError("Implement me")

    @property
    def results(self):
        if not hasattr(self, '_results'):
            self._results = self.make_results()
        return self._results

    def make_results(self):
        """
        Format and return a HypothesisTestResults object
        """
        raise NotImplementedError("Implement me")


class FrequentistProcedure(InferenceProcedure):

    def __init__(self, hypothesis='larger', *args, **kwargs):
        super(FrequentistProcedure, self).__init__(*args, **kwargs)
        self.hypothesis = hypothesis

    def run(self, control_samples, variation_samples,
            alpha=None, inference_kwargs=None):
        raise NotImplementedError("Implement me")

    def make_results(self):
        raise NotImplementedError("Implement me")

    @property
    def control_name(self):
        return self.comparison.d2.name

    @property
    def variation_name(self):
        return self.comparison.d1.name

    @property
    def hypothesis_sm(self):
        """
        Statsmodels-compatible hypothesis
        """
        return self.hypothesis if self.hypothesis != 'unequal' else 'two-sided'

    def accept_hypothesis(self, stat_value):
        """
        Accept the null hypothesis based on the calculated statistic and statitic
        distribution.
        """
        if self.hypothesis == 'larger':
            return stat_value > self.stat_dist.ppf(1 - self.alpha)
        elif self.hypothesis == 'smaller':
            return stat_value < self.stat_dist.ppf(self.alpha)
        elif self.hypothesis == 'unequal':
            return abs(stat_value) > self.stat_dist.ppf(1 - self.alpha / 2.)
        else:
            raise ValueError('Unknown hypothesis: {!r}'.format(self.hypothesis))

    @property
    def hypothesis_text(self):

        control_name = self.control_name if self.control_name else 'control'
        variation_name = self.variation_name if self.variation_name else 'variation'

        if self.hypothesis == 'larger':
            return "{} is larger".format(variation_name)
        elif self.hypothesis == 'smaller':
            return "{} is smaller".format(variation_name)
        elif self.hypothesis == 'unequal':
            return "{} != {}".format(variation_name, control_name)
        else:
            raise ValueError('Unknown hypothesis: {!r}'.format(self.hypothesis))

    @property
    def ci_percents(self):
        if self.hypothesis == 'larger':
            return (self.alpha, np.inf)
        elif self.hypothesis == 'smaller':
            return (-np.inf, 1 - self.alpha)
        elif self.hypothesis == 'unequal':
            return ((self.alpha / 2.), 1 - (self.alpha / 2.))
        else:
            raise ValueError('Unknown hypothesis: {!r}'.format(self.hypothesis))

    @property
    def test_statistic(self):
        return self.comparison.test_statistic
