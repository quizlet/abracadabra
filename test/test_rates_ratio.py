import pytest
from abra import Experiment, HypothesisTest


def test_rates_ratio_larger(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        hypothesis='larger',
        control='A', variation='C'
    )
    ab_results = exp.run_test(ab_test)
    assert ab_results.accept_hypothesis


def test_rates_ratio_smaller(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        hypothesis='smaller',
        control='A', variation='C'
    )
    ab_results = exp.run_test(ab_test)
    assert not ab_results.accept_hypothesis

def test_rates_ratio_unequal(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        hypothesis='unequal',
        control='A', variation='C'
    )
    ab_results = exp.run_test(ab_test)
    assert ab_results.accept_hypothesis

def test_rates_ratio_aa(counts_data):
    exp = Experiment(data=counts_data)
    aa_test = HypothesisTest(
        inference_method='rates_ratio',
        metric='metric',
        control='A', variation='A'
    )
    aa_results = exp.run_test(aa_test)
    assert not aa_results.accept_hypothesis
