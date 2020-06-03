#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Each function returns Stan model code that are appropriate for modeling
distributions of binary or proportions data.
"""


def beta_binomial():
    """
    Return Stan code for for a Beta-Binomial model that compares two samples
    based on their summary statistics. Must provide hyperparameters `alpha_` and
    `beta_` as input data fields.

    Input Data
    ----------
    {
        n_control : int       # the number of control samples
        n_variation : int     # the number of variation samples
        s_control : int       # the number of control succeses
        s_variation : int     # the number of variations successes
        alpha_ : float        # hyperparameter
        beta_ : float         # hyperparameter
    }
    """
    return """
data {
    int<lower=0> n_control; // number of control trials
    int<lower=0> s_control; // number of control successes

    int<lower=0> n_variation; // number of variation trials
    int<lower=0> s_variation; // number of control successes

    real<lower=0> alpha_; // beta prior alpha hyperparameter
    real<lower=0> beta_; // beta prior beta hyperparameter
}

parameters {
    real<lower=0, upper=1> p_control;
    real<lower=0, upper=1> p_variation;
}

transformed parameters {
    real delta;
    real delta_relative;
    real effect_size;

    delta = p_variation - p_control;
    delta_relative = p_variation / p_control - 1.0;
    effect_size = delta / sqrt((p_control * (1 - p_control) + p_variation * (1 - p_variation)) / 2);
}

model {

    // Beta priors on succes rate
    p_control ~ beta(alpha_, beta_);
    p_variation ~ beta(alpha_, beta_);

    // Binomial likelihoods
    s_control ~ binomial(n_control, p_control);
    s_variation ~ binomial(n_variation, p_variation);
}
generated quantities {}

"""


def binomial():
    """
    Returns Stan model code for a Binomial likelihood model with a uniform prior
    """
    return beta_binomial()


def bernoulli():
    """
    Return model code for a Beta-Bernouilli model to be compiled by PyStan.
    Compares two arrays of binary data, where each entry in an array indicates
    success on an associated trial. Must provide hyperparameters `alpha_` and
    `beta_` as input data fields.

    Input Data
    ----------
    {
        n_control : int             # the number of control samples
        n_variation : int           # the number of variation samples
        control : sequence[int]     # the control samples (binary)
        variation : sequence[int]   # the variation samples (binary)
        alpha_ : float              # hyperparameter
        beta_ : float               # hyperparameter
    }
    """
    return """
data {
    int<lower=0> n_control;
    int<lower=0,upper=1> control[n_control];

    int<lower=0> n_variation;
    int<lower=0,upper=1> variation[n_variation];

    real<lower=0> alpha_; // beta prior alpha hyperparameter
    real<lower=0> beta_; // beta prior beta hyperparameter
}

parameters {
    real<lower=0, upper=1> theta_control;
    real<lower=0, upper=1> theta_variation;
}

transformed parameters {
    real delta;
    real delta_relative;
    real effect_size;

    delta = theta_variation - theta_control;
    delta_relative = theta_variation / theta_control - 1.0;
    effect_size = delta / sqrt((theta_control * (1 - theta_control) + theta_variation * (1 - theta_variation)) / 2);
}

model {
    // Beta prior
    theta_control ~ beta(alpha_, beta_);
    theta_variation ~ beta(alpha_, beta_);

    // Bernoulli Likelihoods
    control ~ bernoulli(theta_control);
    variation ~ bernoulli(theta_variation);
}
"""
