#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Function that return Stan model code for modeling data with counts or rates-generated
data.
"""


def gamma_poisson():
    """
    Return Stan model code for a Gamma-Poisson Bayesian model. The model takes
    as inputs two sequences of count variables, with each entry representing the
    total number of target events that occur over a standardized time interval.
    Must provide hyperparameters `alpha_` and `beta_` as input data fields.

    Input Data
    ----------
    {
        n_control : int             # the number of control samples
        n_variation : int           # the number of variation samples
        control : sequence[int]     # the control samples
        variation : sequence[int]   # the variation samples
        alpha_ : float              # hyperparameter
        beta_  : float              # hyperparameter
    }

    """
    return """
data {
    int<lower=0> n_control;
    int control[n_control];

    int<lower=0> n_variation;
    int variation[n_variation];

    real<lower=0> alpha_; // beta prior alpha hyperparameter
    real<lower=0> beta_; // beta prior beta hyperparameter
}

parameters {
    real<lower=0> lambda_control;
    real<lower=0> lambda_variation;
}

transformed parameters {
    real delta;
    real delta_relative;
    real effect_size;

    delta = lambda_variation - lambda_control;
    effect_size = lambda_variation / lambda_control;
    delta_relative = effect_size - 1;

}

model {
    // Gamma priors
    lambda_control ~ gamma(alpha_, beta_);
    lambda_variation ~ gamma(alpha_, beta_);

    // Poisson likelihoods
    control ~ poisson(lambda_control);
    variation ~ poisson(lambda_variation);
}

generated quantities {}
    """
