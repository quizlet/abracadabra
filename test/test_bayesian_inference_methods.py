import pytest
from abra import Experiment, HypothesisTest


@pytest.mark.stan_test
def test_bayesian_gaussian(means_data):
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
def test_bayesian_bernoulli(proportions_data_small):
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
def test_bayesian_binomial(proportions_data_large):
    exp = Experiment(data=proportions_data_large)
    test = HypothesisTest(
        inference_method='binomial', metric='metric',
        control='A', variation='C'
    )
    inference_kwargs = dict(inference_method='sample')
    test_results = exp.run_test(test, inference_kwargs=inference_kwargs)
    assert pytest.approx(test_results.prob_greater, rel=.1, abs=.01) == 1.


@pytest.mark.stan_test
def test_empty_observations_exception(proportions_data_large):
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
