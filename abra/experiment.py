#!/usr/bin/python
# -*- coding: utf-8 -*-
from datetime import datetime
from abra.dataset import Dataset
from abra.stats import Samples, MultipleComparisonCorrection
from abra.mixin import InitRepr
from abra.config import DEFAULT_ALPHA
from abra.hypothesis_test import HypothesisTestSuiteResults
import copy


class Experiment(InitRepr):
    """
    Parameters
    ----------
    data : DataFrame
        the tabular data to analyze, must have columns that correspond
        with `treatment`, `measures`, `attributes`, and `enrollment` if any
        of those are defined.
    treatment : str
        The column in `data` that identifies the association of each
        enrollment in the experiment with one of the experiment conditions. If
        None provided, the global config is searched to identify potential
        treatments in `data`.
    measures : list[str]
        Columns in the dataset that are associated with indicator measurements.
        If None provided, the global config is searched to identify potential
        measures in `data`.
    attributes : list[str]
        the columns in `data` that define segmenting attributes
        associated with each enrollment in the experiment. If None provided, the
        global config is searched to identify potential attributes in `data`.
    name : str
        the name of the experiment
    date : datetime.datetime
        the datetime of the experiment analysis
    enrollment_strategy : str
        metadata for the type of enrollment used to generate the data
    meta : list[str]
        Any additional columns in `data` that should be included in the experiment
        dataset. These columns can be used for additional / custom segmentations.
    """
    __ATTRS__ = ["name", "enrollment_strategy", "date"]

    def __init__(
        self,
        data,
        treatment=None,
        measures=None,
        attributes=None,
        date=None,
        name=None,
        enrollment_strategy=None,
        meta=None
    ):

        # metadata
        if date is None:
            self.date = datetime.now().date()
        elif isinstance(date, datetime):
            self.date = date
        else:
            raise ValueError('date parameter must be a datetime.datetime object')

        self.name = name
        self.enrollment_strategy = enrollment_strategy

        self.dataset = Dataset(data, treatment, measures, attributes, meta)

    @property
    def ds(self):
        """
        Shorthand for the experiment dataset instance
        """
        return self.dataset

    @property
    def measures(self):
        return self.dataset.measures

    @property
    def attributes(self):
        return self.dataset.attributes

    @property
    def observations(self):
        return self.ds.data

    def run_test(
        self,
        test,
        alpha=DEFAULT_ALPHA,
        correction_method=None,
        display_results=False,
        visualize_results=False,
        inference_kwargs=None
    ):
        """
        Given a HypothesisTest, run the test and return the results.

        Parameters
        ----------
        test : HypothesisTest
            A hypothesis test object
        alpha : float
            The Type I error assumed by the experimenter when running the test
        correction_method: str
            Correction method used when running test (if any)
        display_results : boolean
            Whether to print the test results to stdout
        visualize_results : boolean
            Whether to render a visual representation of the results

        Returns
        -------
        test_results: an instance of a HypothesisTestResults or sublass
            The results of the statistical test.
        """

        control_obs = test.filter_variations(self.dataset, test.control, self.ds.treatment)
        control_obs = test.filter_segments(control_obs)
        control_obs = test.filter_metrics(control_obs)
        control_samples = Samples(control_obs, name=test.control)

        variation_obs = test.filter_variations(self.dataset, test.variation, self.ds.treatment)
        variation_obs = test.filter_segments(variation_obs)
        variation_obs = test.filter_metrics(variation_obs)
        variation_samples = Samples(variation_obs, name=test.variation)

        test_results = test.run(
            control_samples,
            variation_samples,
            alpha=alpha, inference_kwargs=inference_kwargs
        )
        test_results.correction_method = correction_method

        if display_results:
            test_results.display()

        if visualize_results:
            test_results.visualize()

        return test_results

    def run_test_suite(
        self,
        test_suite,
        alpha=DEFAULT_ALPHA,
        display_results=False,
        visualize_results=False,
        inference_kwargs=None
    ):
        """
        Given a HypothesisTestSuite, run all tests, perform multiple comparison
        alpha correction, and adjust inference.

        ----------
        test : HypothesisTest
            A hypothesis test object
        alpha : float
            The Type I error assumed by the experimenter when running the test
        display_results : boolean
            Whether to print the test results to stdout
        visualize_results : boolean
            Whether to render a visual representation of the results

        Returns
        -------
        test_results: an instance of a HypothesisTestSuiteResults or subclass
            The results of the statistical test suite.
        """
        corrected_tests = [copy.deepcopy(t) for t in test_suite.tests]

        # run original tests
        original_test_results = [
            self.run_test(test, alpha, inference_kwargs=inference_kwargs)
            for test in test_suite.tests
        ]

        # get p_values for multiple comparison procedure
        p_values = [t.p_value for t in original_test_results]

        correction_method = test_suite.correction_method
        correction = MultipleComparisonCorrection(
            p_values=p_values,
            alpha=alpha,
            method=correction_method
        )

        # rerun tests with updated alpha
        corrected_test_results = []
        for test in corrected_tests:
            corrected_result = self.run_test(
                test,
                alpha=correction.alpha_corrected,
                correction_method=correction.method
            )
            corrected_result.correction_method = correction_method
            corrected_result.render_tables()  # update display params
            corrected_test_results.append(corrected_result)

        return HypothesisTestSuiteResults(
            test_suite.tests,
            original_test_results,
            corrected_test_results,
            correction
        )
