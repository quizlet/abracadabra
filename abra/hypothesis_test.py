#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.mixin import InitRepr, Dataframeable
from abra.utils import ensure_dataframe, run_context
from abra.config import DEFAULT_ALPHA
from abra.inference import InferenceProcedure, get_inference_procedure
from abra.stats import CORRECTIONS
from datetime import datetime
from prettytable import PrettyTable
from collections import OrderedDict
from copy import deepcopy


class CohortFilter(object):
    """
    Filtering interface for selecting cohorts of a dataframe. Powered pandas
    `query` method.

    Parameters
    ----------
    treatment_column : str
        The column associated with various treatments
    treatment_name : str
        When applied, selects out rows with `treatment_column` == `treatment_name`
    """
    def __init__(self, treatment_column, treatment_name):
        self.treatment_column = treatment_column
        self.treatment_name = treatment_name

    def apply(self, data):
        """
        Parameters
        ----------
        data : DataFrame

        Returns
        -------
        cohort_data : DataFrame
            cohort data, filtered from `data`
        """
        return data.query(f"`{self.treatment_column}` == '{self.treatment_name}'")


class SegmentFilter(object):
    """
    Simple filtering interface for selecting subsets of a dataframe. Powered
    pandas `query` interface.
    """
    def __init__(self, segment_pattern):
        self.segment_pattern = segment_pattern

    def apply(self, data):
        if self.segment_pattern is None:
            return data
        else:
            return data.query(self.segment_pattern)


class CustomMetric(object):
    """
    Metric definition that combines one or more measure columns in dataframe.

    Parameters
    ----------
    f: function
        the function that defines an operation performed on each row of the
        dataframe, resulting in a derived value.
    """
    def __init__(self, f):
        self.f = f

    def apply(self, df):
        return df.apply(self.f, axis=1)


class HypothesisTest(InitRepr):
    """
    Executes directives for running inference a procudure to compare the central
    tendencies of two groups of random samples.

    Parameters
    ----------
    inference_method : str
        The name of the inference method used to perform the hypothesis test.
        Can be one of the following:

            Frequentist Inference:
                - 'means_delta'         Continuous
                - 'proprortions_delta'  Proportions
                - 'rates_ratio'         Counts / rates
            Bayesian Inference:
                - 'gaussian'            Continuous
                - 'exp_student_t'       Continuous
                - 'bernoulli'           Proportions / binary
                - 'beta_binomial'       Proportions
                - 'binomial'            Proportions
                - 'gamma_poisson'       Counts / rates

    metric: str or CustomMetric instance (optional)
        a performance indicator over which statistical analyses
        will be performed. Each can either be a measurment in the experiment's
        dataset, or an instance of a CustomMetric. If None provided, the
    control : str
        The name of the control treatment
    variation : str
        The name of the experimental treatment
    segmentation : str or list[str] (optional)
        Defines a list of logical filter operations that follows the conventions
        used by Panda's dataframe query api and segments the treatments into subgroups.
        If a list provided, all operations are combined using logical AND
    date : datetime (optional)
        The datetime the test was run. If None provided, we assume the current
        time.
    suppress_stan_output : boolean
        Whether or not to supress the underlying Stan code optimization stdout.
    **infer_params : dict
        Any additional parameters to be passed to the inference procud
    """
    __ATTRS__ = ['metric', 'control', 'variation', 'segmentation', 'method']

    def __init__(self, inference_method, metric=None,
                 control=None, variation=None,
                 segmentation=None, date=None,
                 suppress_stan_output=True,
                 **infer_params):

        self.metric = metric
        self.control = control
        self.variation = variation
        self.inference_method = inference_method

        self.inference_procedure = get_inference_procedure(inference_method, **infer_params)
        self.suppress_stan_output = suppress_stan_output

        if isinstance(segmentation, list):
            segmentation = " & ".join(segmentation)

        self.segmentation = segmentation

        if date is None:
            self.date = datetime.now().date()

        if isinstance(self.metric, CustomMetric):
            self.metric_column = self.metric.f.__name__
        else:
            self.metric_column = metric

    def _add_custom_metric_column(self, _data):
        data = _data.copy()
        self.metric_name = self.metric.f.__name__
        data[self.metric_name] = self.metric.apply(data)
        return data

    def filter_variations(self, data, variation_name, treatment=None):
        """
        Helper function used to pull out the data series observations for a
        variation.
        """
        data = ensure_dataframe(data)
        if isinstance(self.metric, CustomMetric):
            data = self._add_custom_metric_column(data)
        else:
            self.metric_name = self.metric

        if treatment is None:
            if hasattr(self, 'treatment'):
                treatment = self.treatment
            else:
                raise ValueError("Cant't determine the treatment column, ",
                                 "please provide as `treatment` argument")

        cohort_filter = CohortFilter(treatment, variation_name)
        return cohort_filter.apply(data)

    def filter_segments(self, data):
        """
        Helper function used to filter observations for all segments.
        """
        data = ensure_dataframe(data)
        segment_filter = SegmentFilter(self.segmentation)
        return segment_filter.apply(data)

    def filter_metrics(self, data):
        """
        Helper function used to filter out observations that have invalid metric
        values.
        """
        return data[data.notna()[self.metric_name]][self.metric_name]

    def run(self, control_samples, variation_samples, alpha=DEFAULT_ALPHA, inference_kwargs=None):
        """
        Run the statistical test inference procedure comparing two groups of
        samples.

        Parameters
        ----------
        control_samples : an instance of `Samples` class
            The samples for the control treatment group
        variation_samples : an instance of `Samples` class
            The samples for the experimental treatment group
        alpha : float in (0, 1) (optional)
            Is either:
                - the 'significance level' for frequentist tests
                - the one minus the credible mass of the posterior over differences
                  between the two groups.
            If None provided, we assume an alpha = 0.05
        inference_kwargs : dict
            Any additional keyword args to be provided to the inference procedure

        Returns
        -------
        results : subclass of `TestResultsBase`
            The container class holding the summary of the experiment

        """
        if not issubclass(self.inference_procedure.__class__, InferenceProcedure):
            raise ValueError(
                "The inference procedure used must subclass `InferenceProcedure`"
            )
        self.alpha = alpha

        # when running Bayesian inference, optionally run with context
        # manager to supress stdout during Stan optimization routines
        supress_stdout = self.inference_procedure.__class__.__name__ == 'BayesianDelta' and self.suppress_stan_output
        with run_context(supress_stdout):
            self.inference_procedure.run(control_samples, variation_samples, alpha, inference_kwargs=inference_kwargs)

        results = self.inference_procedure.make_results()
        results.metric_name = self.metric_column
        results.segmentation = self.segmentation
        return results

    def copy(self, **kwargs):
        copy = deepcopy(self)
        infer_kwargs = kwargs.get("infer_kwargs", {})
        for k, v in kwargs.items():
            if hasattr(copy, k):
                setattr(copy, k, v)

        copy.inference_procedure = get_inference_procedure(copy.inference_method, **infer_kwargs)
        return copy


class TestResultsBase(Dataframeable):

    def display(self):
        raise NotImplementedError("Implement me")

    def visualize(self, figsize=None, outfile=None, *args, **kwargs):
        raise NotImplementedError("Implement me")

    def to_csv(self, fname, delimiter=','):
        """
        Export result to delimiter-separated value file
        """
        results_df = self.to_dataframe()
        results_df.to_csv(fname, sep=delimiter, encoding='utf8', index=False)

    @property
    def json(self):
        raise NotImplementedError('Must implement json property')


class HypothesisTestResults(TestResultsBase):
    """
    Base class for all hypothesis test results

    Parameters
    ----------
    control : Samples
        the control samples
    variation : Samples
        the variation samples
    metric : str
        The name of the metric being measured
    delta: float
        the absolute difference between the variation and control sample means
    delta_relative: float
        the percent difference between the variation and control sample means
    inference_procedure: subclass of InferenceProcedure
        the inference procedure used to generate the hypothesis test results
    metric_name : str
        The name of the metric that is being compared
    warnings : list[str]
        A list of any warning messages accumultaed during the test
    aux : dict
        Auxillary variables used for displaying or visualizing specific types
        of tests.
    """
    def __init__(self, control, variation,
                 delta, delta_relative, effect_size,
                 inference_procedure, warnings=[], aux={}):

        self.control = control
        self.variation = variation
        self.delta = delta
        self.delta_relative = delta_relative
        self.effect_size = effect_size
        self.inference_procedure = inference_procedure
        self.metric_name = None  # this gets updated by the test
        self.warnings = warnings
        self.aux = aux

    @property
    def samples_table(self):
        if not hasattr(self, "_samples_table"):
            self.render_samples_table()
        return self._samples_table

    @property
    def stats_table(self):
        if not hasattr(self, "_stats_table"):
            self.render_stats_table()
        return self._stats_table

    @property
    def table(self):
        if not hasattr(self, "_results_table"):
            self.render_tables()
        return self._results_table

    def render_samples_table(self):
        tbl = PrettyTable(header=True)
        tbl.add_column(
            "Treatment",
            [
                "Metric",
                "Observations",
                "Mean",
                "Standard Error",
                "Variance"
            ],
            align='l'
        )

        tbl.add_column(
            self.control.name,
            [
                self.metric_name,
                int(self.control.nobs),
                f"{self.control.mean:1.4f}",
                "{!r}".format(tuple([round(s, 4) for s in self.control.std_err(self.alpha)])),
                f"{self.control.var:1.4f}"
            ],
            align='l'
        )

        tbl.add_column(
            self.variation.name,
            [
                self.metric_name,
                int(self.variation.nobs),
                f"{self.variation.mean:1.4f}",
                "{!r}".format(tuple([round(s, 4) for s in self.variation.std_err(self.alpha)])),
                f"{self.variation.var:1.4f}"
            ],
            align='l'
        )
        self._samples_table = str(tbl)

    def __repr__(self):
        return self.table

    def render_tables(self):
        self.render_stats_table()
        self.render_samples_table()
        self._results_table = f"""
Observations Summary:
{self.samples_table}

Test Results:
{self.stats_table}
        """

    @property
    def _base_json(self):
        """
        Parameters that belong to all hypothesis test results
        """
        return OrderedDict([('metric', [self.metric_name]),
                            ('hypothesis', [self.hypothesis]),
                            ('model_name', [self.model_name]),
                            ('accept_hypothesis', [self.accept_hypothesis]),

                            ('control_name', [self.control.name]),
                            ('control_nobs', [self.control.nobs]),
                            ('control_mean', [self.control.mean]),
                            ('control_ci', [self.control.ci(self.alpha)]),
                            ('control_var', [self.control.var]),

                            ('variation_name', [self.variation.name]),
                            ('variation_nobs', [self.variation.nobs]),
                            ('variation_mean', [self.variation.mean]),
                            ('variation_ci', [self.variation.ci(self.alpha)]),
                            ('variation_var', [self.variation.var]),

                            ('delta', [self.delta]),
                            ('delta_relative', [100 * self.delta_relative]),
                            ('effect_size', [self.effect_size]),
                            ('alpha', [self.alpha]),
                            ('segmentation', [self.segmentation]),
                            ('warnings', [process_warnings(self.warnings)])
                            ])

    def display(self):
        raise NotImplementedError

    def visualize(self, figsize=None, outfile=None, *args, **kwargs):
        raise NotImplementedError


def process_warnings(warnings):
    if isinstance(warnings, list):
        warnings = ';'.join(warnings)
    return warnings


class HypothesisTestSuite(object):
    """
    Parameters
    ----------
    tests : list[HypothesisTest]
        The tests to run
    correction_method : str
        One of the following correction methods:
            'bonferroni', 'b' : one-step Bonferroni correction
            'sidak', 's' : one-step Sidak correction
            'fdr_bh', 'bh; : Benjamini/Hochberg (non-negative)
    """
    def __init__(self, tests, correction_method='sidak'):

        if not isinstance(tests, (list, tuple)):
            raise ValueError('`tests` must be a sequence of `HypothesisTest` instances')

        for test in tests:
            if not issubclass(test.__class__, HypothesisTest):
                raise ValueError("All tests must subclass `HypothesisTest`")

        if correction_method not in set(list(CORRECTIONS.keys()) + list(CORRECTIONS.values())):
            raise ValueError(f'Correction method {correction_method} not supported')

        self.tests = tests
        self.correction_method = CORRECTIONS[correction_method] \
                                 if correction_method in CORRECTIONS \
                                 else correction_method


class HypothesisTestSuiteResults(TestResultsBase):
    """
    Store, display, visualize, and export the results of statistical test
    suite.
    """
    def __init__(self, tests, original_results, corrected_results, correction):
        self.tests = tests
        self.ntests = len(tests)
        self.original_results = original_results
        self.corrected_results = corrected_results
        self.correction = correction

    def __repr__(self):
        return f"{self.__class__.__name__}(ntests={self.ntests}, correction_method='{self.correction.method}')"

    def display(self):
        for ii, res in enumerate(self.corrected_results):
            print('-' * 60)
            print(f'Test {ii + 1} of {self.ntests}')
            print(res)

    def visualize(self, figsize=None, outfile=None, *args, **kwargs):
        for ii, res in enumerate(self.corrected_results):
            res.visualize()
