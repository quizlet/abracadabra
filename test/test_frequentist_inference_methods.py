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


def test_small_proportions_delta_expermiment(proportions_data_small):
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


def test_means_delta_experiment(means_data):
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


def test_rates_ratio_experiment(counts_data):
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
