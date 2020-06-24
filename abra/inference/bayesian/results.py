#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.hypothesis_test import HypothesisTestResults, PrettyTable, OrderedDict


class BayesianTestResults(HypothesisTestResults):
    """
    Class for storing Bayesian hypothesis test statistics.

    Parameters
    ----------
    alpha : float in (0, 1)
        The "significance" level, or type I error rate for the experiment
    traces : instance of abra.vis.Traces
        The results of a Bayesian inference procedure.
    hdi : tuple
        The values and percentiles associated with the lower and upper bounds of
        the highest density interval around delta posterior, based on `alpha`
    model_name : str
        The name of the Bayesian model used.
    hypothesis : str
        Human-readable message for interpreting the experiment
    inference_method : str
        The name of the simulation method used (e.g. 'sampling', 'variational')
    data_type : python type object
        The data type of the observations used in the inference,
    hyperparameters : dict:
        Hyperparmas that specify specify model
    """
    def __init__(self, alpha, traces, hdi, hdi_relative, prob_greater, model_name,
                 hypothesis, inference_method, data_type, warnings=None, hyperparameters=None,
                 *args, **kwargs):

        super(BayesianTestResults, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.traces = traces
        self.hypothesis = hypothesis
        self.hdi = hdi
        self.hdi_relative = hdi_relative
        self.prob_greater = prob_greater
        self.model_name = model_name
        self.hypothesis = hypothesis
        self.inference_method = inference_method
        self.data_type = data_type
        self.hyperparameters = hyperparameters
        # self.accept_hypothesis = (1 - self.prob_greater) < self.alpha   # this is VERY conservative
        self.accept_hypothesis = self.prob_greater >= 1 - self.alpha  # cutoff at 75% chance p(greater)
        self.warnings = "\n".join(warnings) if warnings else None

    def render_stats_table(self):
        tbl = PrettyTable(header=False)
        tbl.add_column(
            "",
            [
                "Delta",
                "HDI",
                "HDI %-tiles",
                "Delta-relative",
                "HDI-relative",
                "Effect Size",
                "alpha",
                "Credible Mass",
                f"p({self.variation.name} > {self.control.name})",
                "Inference Method",
                "Model Hyperarameters",
                "Inference Method",
                "Hypothesis",
                "Accept Hypothesis",
                "Warnings"
            ],
            align="c"
        )
        tbl.add_column(
            "",
            [
                "{:1.4f} ".format(self.delta),
                "({:1.4f}, {:1.4f})".format(self.hdi[0][0], self.hdi[0][1]),
                "({:1.4f}, {:1.4f})".format(self.hdi[1][0], self.hdi[1][1]),
                "{:2.2f} %".format(100 * self.delta_relative),
                "({:1.2f}, {:1.2f}) %".format(
                    100 * self.hdi_relative[0][0],
                    100 * self.hdi_relative[0][1]
                ),
                "{:1.4f}".format(self.effect_size),
                "{:1.4f}".format(self.alpha),
                "{:1.4f}".format(1 - self.alpha),
                "{:1.4f}".format(self.prob_greater),
                "{!r}".format(self.model_name),
                "{!r}".format(self.hyperparameters),
                "{!r}".format(self.inference_method),
                "{!r}".format(self.hypothesis),
                "{!r}".format(self.accept_hypothesis),
                "{!r}".format(self.warnings)
            ],
            align="l"
        )
        self._stats_table = str(tbl)

    @property
    def json(self):
        """
        Add results properties that are special to Bayesian test
        """
        _json = self._base_json
        _json.update(OrderedDict([('test_type', ['bayesian']),
                                  ('p', [self.prob_greater]),
                                  ('p_interpretation', ["p(variation > control)"]),
                                  ('delta_ci', [self.hdi[0]]),
                                  ('ntiles_ci', [self.hdi[1]]),
                                  ('delta_relative_ci', [(100 * self.hdi_relative[0][0],
                                                          100 * self.hdi_relative[0][1])]),
                                  ('ci_interpretation', ['Highest Density Interval']),

                                  ('delta_hdi', [self.hdi[0]]),
                                  ('ntiles_hdi', [self.hdi[1]]),
                                  ('delta_relative_hdi', [(100 * self.hdi_relative[0][0],
                                                           100 * self.hdi_relative[0][1])]),
                                  ('credible_mass', [(1 - self.alpha)]),
                                  ('inference_method', [self.inference_method]),
                                  ('prob_greater', [self.prob_greater])]))
        return _json

    def display(self):
        print(self)

    def visualize(self, figsize=None, outfile=None, *args, **kwargs):
        from abra.vis import visualize_bayesian_results, RESULTS_FIGSIZE
        figsize = figsize if figsize else RESULTS_FIGSIZE
        visualize_bayesian_results(self, figsize, outfile, *args, **kwargs)
