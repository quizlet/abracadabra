"""
Each function returns Stan model code that are appropriate for modeling data with
continuous, symmetric distributions.
"""


def gaussian():
    """
    Return Stan code for Gaussian model with Uniform priors on variance
    and Gaussian priors on the mean. Must provide hyperparameter `std_max` as
    input data field.

    Input Data
    -----------
    {
        n_control : int               # the number of control samples
        n_variation : int             # the number of variation samples
        control : sequence[float]     # the control samples
        variation : sequence[float]   # the variation samples
        std_max : float               # hyperparameter
    }
    """
    return """
data {
    int<lower=0> n_control;
    int<lower=0> n_variation;

    real control[n_control];
    real variation[n_variation];

    real<lower=0> std_max;
}

parameters {
    real mu_control;
    real mu_variation;

    real<lower=0, upper=std_max> sigma_control;
    real<lower=0, upper=std_max> sigma_variation;
}

transformed parameters {
    real delta;
    real delta_relative;
    real effect_size;

    real mean_control;
    real mean_variation;

    real<lower=0, upper=std_max> std_control;
    real<lower=0, upper=std_max> std_variation;

    mean_control = mean(control);
    mean_variation = mean(variation);

    std_control = sd(control);
    std_variation = sd(variation);

    delta = mu_variation - mu_control;
    delta_relative = (mu_variation / mu_control) - 1.0;
    effect_size = delta / sqrt((sigma_control^2 + sigma_variation^2) / 2);
}

model {
    // Gaussian priors on means
    mu_control ~ normal(mean_control, std_control);
    mu_variation ~ normal(mean_variation, std_variation);

    // Uniform priors on standard deviations
    sigma_control ~ uniform(0, std_max);
    sigma_variation ~ uniform(0, std_max);

    // Gaussian Likelihood
    control ~ normal(mu_control, sigma_control);
    variation ~ normal(mu_variation, sigma_variation);
}
generated quantities {}
    """


def exp_student_t():
    """
    Return Stan model code for a Student's t Bayesian model with an Exponential
    prior on the degrees of freedom, Gaussian priors on the location, and positive
    Uniform prior on the scale inverse precision of the distribution. Must provide
    hyperparameters `precision` and `std_max` as input data fields.

    The Student's t model will generally be more robust to large outliers than
    the Gaussian model

    Input Data
    ----------
    {
        n_control : int               # the number of control samples
        n_variation : int             # the number of variation samples
        control : sequence[float]     # the control samples
        variation : sequence[float]   # the variation samples
        mean_control : float          # the sample mean for control samples
        mean_variation : float        # sample mean for variation samples
        std_control : float           # the control sample std dev
        std_variation : float         # the variation sample std dev
        precision : float             # hyperparameter
        std_max : float               # hyperparameter
    }

    """
    return """
data {
    int<lower=0> n_control;
    real control[n_control];

    int<lower=0> n_variation;
    real variation[n_variation];

    real<lower=0> std_max;
    real<lower=1> precision;
}

parameters {
    real mu_control;
    real<lower=0> sigma_control;

    real mu_variation;
    real<lower=0> sigma_variation;

    real eta;
}

transformed parameters {
    real mean_control;
    real mean_variation;

    real<lower=0, upper=std_max> std_control;
    real<lower=0, upper=std_max> std_variation;

    real lambda_control;
    real lambda_variation;
    real delta;
    real delta_relative;
    real effect_size;

    mean_control = mean(control);
    mean_variation = mean(variation);

    std_control = sd(control);
    std_variation = sd(variation);

    // student t requires precisions
    lambda_control = 1 / sigma_control^2;
    lambda_variation = 1 / sigma_variation^2;

    delta = mu_variation - mu_control;
    delta_relative = (mu_variation / mu_control) - 1.0;
    effect_size = delta / sqrt((sigma_control^2 + sigma_variation^2) / 2);
}

model {
    // Gaussian priors on mean
    mu_control ~ normal(mean_control, std_control);
    mu_variation ~ normal(mean_variation, std_variation);

    // Uniform prior on variance
    sigma_control ~ uniform(0, std_max);
    sigma_variation ~ uniform(0, std_max);

    // Exponential prior on student's degrees of freedom
    eta ~ exponential(1 / precision);

    // Student-t likelihoods
    control ~ student_t(eta, mu_control, lambda_control);
    variation ~ student_t(eta, mu_variation, lambda_variation);
}
generated quantities {}
    """
