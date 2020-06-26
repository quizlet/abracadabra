import pytest
from abra import Experiment, HypothesisTest


def test_large_proportions_delta_expermiment(proportions_data_large):
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


def test_proportions_delta_ab_unequal(proportions_data_small):
    exp = Experiment(proportions_data_small, name='proportions-test')

    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='unequal',
        inference_method='proportions_delta'
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic == 'z'
    assert results_ab.accept_hypothesis

def test_proportions_delta_ab_larger(proportions_data_small):
    exp = Experiment(proportions_data_small, name='proportions-test')

    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='larger',
        inference_method='proportions_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.accept_hypothesis

def test_proportions_delta_ab_smaller(proportions_data_small):
    exp = Experiment(proportions_data_small, name='proportions-test')

    # run A/B test
    test_ab = HypothesisTest(
        metric='metric',
        control='A', variation='F',
        hypothesis='smaller',
        inference_method='proportions_delta'
    )
    results_ab = exp.run_test(test_ab)
    assert not results_ab.accept_hypothesis


def test_proportions_delta_aa(proportions_data_small):
    exp = Experiment(proportions_data_small, name='proportions-test')

    # run A/A test
    test_aa = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='larger',
        inference_method='proportions_delta'
    )
    results_aa = exp.run_test(test_aa)
    assert not results_aa.accept_hypothesis


def test_proportions_delta_experiment_t(proportions_data_small):
    """Small sample sizes defautl to t-tests"""
    exp = Experiment(proportions_data_small.sample(29), name='proportions-test')

    test_aa = HypothesisTest(
        metric='metric',
        control='A', variation='A',
        hypothesis='unequal',
        inference_method='means_delta'
    )
    results_aa = exp.run_test(test_aa)
    assert results_aa.test_statistic == 't'