import pytest
from abra import Experiment, HypothesisTest


def test_means_delta_experiment_t(means_data):
    """Small sample sizes defautl to t-tests"""
    exp = Experiment(means_data.sample(29), name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='unequal',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic == 't'

def test_means_delta_experiment_unequal_ab(means_data):
    exp = Experiment(means_data, name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='unequal',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_larger_ab(means_data):
    exp = Experiment(means_data, name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='larger',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_smaller_ab(means_data):
    exp = Experiment(means_data, name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='smaller',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic == 'z'
    assert not results_ab.accept_hypothesis


def test_means_delta_experiment_aa(means_data):
    exp = Experiment(means_data, name='means-test')

    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='larger',
        inference_method='means_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic == 'z'
    assert not results_ab.accept_hypothesis