#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from abra import (
    Experiment,
    HypothesisTest,
    HypothesisTestSuite,
    MultipleComparisonCorrection,
    CustomMetric
)
from abra.utils import generate_fake_observations

proportions_data_large = generate_fake_observations(
    distribution='bernoulli',
    n_treatments=3,
    n_attributes=4,
    n_observations=10000
)

proportions_data_small = generate_fake_observations(
    distribution='bernoulli',
    n_treatments=6,
    n_observations=6 * 50
)

means_data = generate_fake_observations(
    distribution='gaussian',
    n_treatments=6,
    n_observations=6 * 50
)


counts_data = generate_fake_observations(
    distribution='poisson',
    n_treatments=3,
    n_observations=3 * 100
)


def test_multiple_comparison():
    p_values = np.arange(.001, .1, .01)
    mc = MultipleComparisonCorrection(p_values, method='b')

    assert mc.alpha_corrected < mc.alpha_orig
    assert mc.accept_hypothesis[0]
    assert not mc.accept_hypothesis[-1]


def test_large_proportions_expermiment():
    exp = Experiment(proportions_data_large, name='proportions-test')

    # run 'A/A' test
    test_aa = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='larger',
        inference_method='proportions_delta'
    )
    results_aa = exp.run_test(test_aa)

    assert results_aa.test_statistic == 'z'
    assert not results_aa.accept_hypothesis

    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='B',
        hypothesis='larger',
        inference_method='proportions_delta'
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis


def test_small_proportions_expermiment():
    exp = Experiment(proportions_data_small, name='proportions-test')

    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='unequal',
        inference_method='proportions_delta'
    )
    results_ab = exp.run_test(test_ab)
    results_ab.to_dataframe()

    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis


def test_means_experiment():
    exp = Experiment(means_data, name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='larger',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    results_ab.to_dataframe()

    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis


def test_counts_experiment():
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        control='A', variation='C'
    )
    ab_results = exp.run_test(ab_test)

    assert ab_results.accept_hypothesis

    aa_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        control='A', variation='A'
    )
    aa_results = exp.run_test(aa_test)

    assert not aa_results.accept_hypothesis


def test_hypothesis_test_suite():
    exp = Experiment(proportions_data_large, name='test')

    # run 'A/A' test (should never reject null)
    test_aa = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='larger',
        inference_method='proportions_delta',
    )
    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='B',
        hypothesis='larger',
        inference_method='proportions_delta'
    )
    test_suite = HypothesisTestSuite([test_aa, test_ab], correction_method='b')
    test_suite_results = exp.run_test_suite(test_suite)

    assert test_suite_results.corrected_results[0].correction_method == 'bonferroni'
    assert test_suite_results.original_results[0].correction_method is None

    # corrected alpha should be smaller
    alpha_orig = test_suite_results.original_results[0].alpha
    alpha_corrected = test_suite_results.corrected_results[0].alpha

    assert alpha_orig > alpha_corrected
    assert not test_suite_results.corrected_results[0].accept_hypothesis
    assert test_suite_results.corrected_results[1].accept_hypothesis


def test_frequentist_methods():
    exp = Experiment(data=proportions_data_large)
    test = HypothesisTest(
        metric='metric',
        control='A', variation='C',
        inference_method='proportions_delta'
    )
    test_results = exp.run_test(test)

    assert test_results.accept_hypothesis


@pytest.mark.stan_test
def test_bayesian_gaussian():
    exp = Experiment(data=means_data)
    test = HypothesisTest(
        inference_method='gaussian',
        metric='metric',
        control='A', variation='F'
    )
    inference_kwargs = dict(inference_method='sample')
    test_results = exp.run_test(test, inference_kwargs=inference_kwargs)
    assert pytest.approx(test_results.prob_greater, rel=.1, abs=.01) == 1.


@pytest.mark.stan_test
def test_bayesian_bernoulli():
    exp = Experiment(data=proportions_data_small)
    test = HypothesisTest(
        inference_method='bernoulli',
        metric='metric',
        control='A', variation='F'
    )
    inference_kwargs = dict(inference_method='sample')
    test_results = exp.run_test(test, inference_kwargs=inference_kwargs)
    assert pytest.approx(test_results.prob_greater, rel=.1, abs=.01) == 1.


@pytest.mark.stan_test
def test_bayesian_binomial():
    exp = Experiment(data=proportions_data_large)
    test = HypothesisTest(
        inference_method='binomial', metric='metric',
        control='A', variation='C'
    )
    inference_kwargs = dict(inference_method='sample')
    test_results = exp.run_test(test, inference_kwargs=inference_kwargs)
    assert pytest.approx(test_results.prob_greater, rel=.1, abs=.01) == 1.


@pytest.mark.stan_test
def test_empty_observations_exception():
    exp = Experiment(data=proportions_data_large)
    # no F variation in proportions_data_large
    with pytest.raises(ValueError):
        test = HypothesisTest(
            inference_method='binomial',
            metric='metric',
            control='A', variation='F'
        )
        inference_kwargs = dict(inference_method='sample')
        _ = exp.run_test(test, inference_kwargs=inference_kwargs)


def test_custom_metric():
    exp = Experiment(means_data, name='means-test')

    def custom_metric(row):
        return 4 + np.random.rand() if row['treatment'] != 'A' else np.random.rand()

    test_ab = HypothesisTest(
        metric=CustomMetric(custom_metric),
        control='A', variation='F',
        hypothesis='larger',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    results_ab.to_dataframe()

    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis
