#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.hypothesis_test import HypothesisTestResults, PrettyTable, OrderedDict
import numpy as np


class FrequentistTestResults(HypothesisTestResults):
    """
    Class for holding, displaying, and exporting frequentist hypothesis test
    results.

    alpha : float in (0, 1)
        The "significance" level, or type I error rate for the experiment
    power : float in (0, 1)
        The statistical power of the experiment, or the probability of correctly
        rejecting the null hypothesis.
    confidence_interval : tuple
        The values and percentiles associated with the lower and upper bounds of
        the confidence interval around delta, based on `alpha`
    test_statistic : str
        The name of the test statistic used to calculate the p-value for the test
    statistic_value : float
        The value of the test statistic used to calculate the p-value
    p_value : float in (0, 1)
        The p-value, based on the test value of `statistic_value`
    df : int
        The degrees of freedom for the experiment, if applicable
    hypothesis : str
        Human-readable message for interpreting the experiment
    accept_hypothesis : boolean
        Whether or not to accept the alternative hypothesis, based on `p_value` and `alpha`
        and any correction method.
    correction_method : str
        The name of the multiple comparison correction method used, if any
    """
    def __init__(self, alpha, confidence_interval,
                 p_value, hypothesis, accept_hypothesis, power=np.nan, df=np.nan,
                 test_statistic=None, statistic_value=np.nan,
                 correction_method=None,
                 warnings=None, *args, **kwargs):

        super(FrequentistTestResults, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.power = power
        self.ci = confidence_interval
        self.test_statistic = test_statistic
        self.statistic_value = statistic_value
        self.p_value = p_value
        self.df = df
        self.hypothesis = hypothesis
        self.accept_hypothesis = accept_hypothesis
        self.correction_method = correction_method
        self.model_name = self.inference_procedure.method
        self.comparison_type = self.inference_procedure.__class__.__name__
        self.estimate_relative_confidence_interval()
        self.warnings = "\n".join(warnings) if warnings else None

    def estimate_relative_confidence_interval(self):
        """
        Estimate the confidence on the relative difference using interpolation.
        """
        self.ci_relative = [
            tuple((self.ci[0] + self.control.mean) / self.control.mean - 1.),
            self.ci[1]
        ]

    def render_stats_table(self):
        alpha_string = "alpha (corrected)" if self.correction_method else "alpha"
        tbl = PrettyTable(header=False)
        tbl.add_column(
            "",
            [
                "{}".format(self.comparison_type),
                "{} CI".format(self.comparison_type),
                "CI %-tiles",
                "{}-relative".format(self.comparison_type),
                "CI-relative",
                "Effect Size",
                alpha_string,
                "Power",
                "Inference Method",
                "Test Statistic ({!r})".format(self.test_statistic),
                "p-value",
                "Degrees of Freedom",
                "Hypothesis",
                "Accept Hypothesis",
                "MC Correction",
                "Warnings",
            ],
            align="c"
        )
        tbl.add_column(
            "",
            [
                "{:1.4f} ".format(self.delta),
                "({:1.4f}, {:1.4f})".format(self.ci[0][0], self.ci[0][1]),
                "({:1.4f}, {:1.4f})".format(self.ci[1][0], self.ci[1][1]),
                "{:2.2f} %".format(100 * self.delta_relative),
                "({:1.2f}, {:1.2f}) %".format(
                    100 * self.ci_relative[0][0], 100 * self.ci_relative[0][1]
                ),
                "{:1.4f}".format(self.effect_size),
                "{:1.4f}".format(self.alpha),
                "{:1.4f}".format(self.power),
                "{!r}".format(self.model_name),
                "{:1.4f}".format(self.statistic_value),
                "{:1.4f}".format(self.p_value),
                "{!r}".format(self.df),
                "{!r}".format(self.hypothesis),
                "{!r}".format(self.accept_hypothesis),
                "{!r}".format(self.correction_method),
                "{!r}".format(self.warnings)
            ],
            align="l"
        )
        self._stats_table = str(tbl)

    def display(self):
        print(self)

    @property
    def json(self):
        """
        Add results properties that are special to Bayesian test
        """
        _json = self._base_json
        _json.update(
            OrderedDict(
                [
                    ('test_type', ['frequentist']),
                    ('p', [self.p_value]),
                    ('p_interpretation', ["p-value"]),
                    ('delta_ci', [self.ci[0]]),
                    ('ntiles_ci', [self.ci[1]]),
                    ('delta_relative_ci', [(100 * self.ci_relative[0][0], 100 * self.ci_relative[0][1])]),
                    ('ci_interpretation', ['Confidence Interval']),
                    ('p_value', [self.p_value]),
                    ('power', [self.power]),
                    ('statistic_name', [self.test_statistic]),
                    ('statistic_value', [self.statistic_value]),
                    ('df', [self.df]),
                    ('mc_correction', [self.correction_method])
                ]
            )
        )
        return _json

    def visualize(self, figsize=None, outfile=None, *args, **kwargs):
        # lazy import
        from abra.vis import (
            visualize_gaussian_results,
            visualize_binomial_results,
            visualize_rates_results,
            visualize_bootstrap_results,
            RESULTS_FIGSIZE
        )

        figsize = figsize if figsize else RESULTS_FIGSIZE
        _model_name = self.model_name.replace(" ", '').replace("-", "").replace("_", "")
        if _model_name in ('meansdelta'):
            visualize_gaussian_results(self, figsize, outfile, *args, **kwargs)
        elif _model_name in ('proportionsdelta'):
            visualize_binomial_results(self, figsize, outfile, *args, **kwargs)
        elif _model_name in ('ratesratio'):
            visualize_rates_results(self, figsize, outfile, *args, **kwargs)
        elif _model_name in ('bootstrap'):
            visualize_bootstrap_results(self, figsize, outfile, *args, **kwargs)
