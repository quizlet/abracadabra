import pytest
from abra import Experiment, HypothesisTest
from abra.inference.bayesian import get_stan_model_code
from abra.inference.bayesian.models.binary import beta_binomial, binomial, bernoulli
from abra.inference.bayesian.models.continuous import gaussian, exp_student_t
from abra.inference.bayesian.models.counts import gamma_poisson


def test_binary_model_specs():
    "Can we better test compilation here?"
    assert isinstance(binomial(), str)
    assert isinstance(bernoulli(), str)
    assert binomial() == get_stan_model_code('binomial')
    assert beta_binomial() == binomial()


def test_continuous_model_specs():
    "Can we better test compilation here?"
    assert isinstance(gaussian(), str)
    assert gaussian() == get_stan_model_code('gaussian')
    assert isinstance(exp_student_t(), str)
    assert exp_student_t() == get_stan_model_code('exp_student_t')


def test_counts_model_specs():
    "Can we better test compilation here?"
    assert isinstance(gamma_poisson(), str)
    assert gamma_poisson() == get_stan_model_code('gamma_poisson')


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
