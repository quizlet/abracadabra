import numpy as np
from abra.experiment import Experiment
from abra.hypothesis_test import (
    HypothesisTest,
    HypothesisTestSuite,
    CustomMetric
)


def test_hypothesis_test_suite(proportions_data_large):
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


def test_custom_metric(means_data):
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

