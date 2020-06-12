# ✨ABracadabra✨
![MIT License](https://img.shields.io/github/license/quizlet/abracadabra)
[![quizlet](https://circleci.com/gh/quizlet/abracadabra.svg?style=shield)](https://circleci.com/gh/circleci/circleci-docs)
![Coverage](https://codecov.io/gh/quizlet/abracadabra/branch/master/graph/badge.svg)


✨ABracadabra✨ is a Python framework consisting of statistical tools and a convenient API specialized for running hypothesis tests on observational experiments (aka “AB Tests” in the tech world). The framework has driven [Quizlet](https://quizlet.com)’s experimentation pipeline since 2018.

## Features
- Offers a simple and intuitive, yet powerful API for running, visualizing, and interpreting statistically-rigorous hypothesis tests with none of the hastle of jumping between various statistical or visualization packages.
- Supports most common variable types used in AB Tests inlcuding:
    + Continuous
    + Binary/Proportions
    + Counts/Rates
- Implements many Frequentist and Bayesian inference methods including:


| Variable Type | Model Class| `inference_method` parameter  |
|---|---|---|
| Continuous | Frequentist| `'means_delta'` (t-test) |
|  | Bayesian| `'gaussian'`, `'exp_student_t'`|
| Binary / Proportions | Frequentist| `'proportions_delta'` (z-test) |
|  | Bayesian| `'beta'`, `'beta_binomial'`, `'bernoulli'`  |
| Counts/Rates  |Frequentist| `'rates_ratio'`
|  |Bayesian| `'gamma_poisson'`  |

- Supports multiple customizations:
    + Custom metric definitions
    + Bayesian priors
    + Easily extendable to support new inference methods


## Installation

### Requirements
- ✨ABracadabra✨ has been tested on `python>=3.7`.

### Install via `pip`
#### from the PyPI index (recommended)

```bash
pip install abracadabra
```

#### from Quizlet's Github repo

```bash
pip install git+https://github.com/quizlet/abracadabra.git
```

### Install from source
If you would like to contribute to ✨ABracadabra✨, then you'll probably want to install from source (or use the `-e` flag when installing from `PyPI`):

```bash
mkdir /PATH/TO/LOCAL/ABRACABARA && cd /PATH/TO/LOCAL/ABRACABARA
git clone git@github.com:quizlet/abracadabra.git
cd abracadabra
python setup.py develop
```

## ✨ABracadabra✨ Basics

### Observations data
✨ABracadabra✨ takes as input a [pandas](https://pandas.pydata.org/) `DataFrame` containing experiment observations data. Each record represents an observation/trial recorded in the experiment and has the following columns:

- **One or more `treatment` columns**: each treatment column contains two or more distinct, discrete values that are used to identify the different groups in the experiment
- **One or more `metric` columns**: these are the values associated with each observation that are used to compare groups in the experiment.
- **Zero or more `attributes` columns**: these are associated with additional properties assigned to the observations. These attributes can be used for any additional segmentations across groups.

To demonstrate, let's generate some artificial experiment observations data. The `metric` column in our dataset will be a series of binary outcomes (i.e. `True`/`False`, here stored as `float` values). This binary `metric` is analogous to *conversion* or *success* in AB testing. These outcomes are simulated from three different Bernoulli distributions, each associated with the `treatement`s named `"A"`, `"B"`, and `"C"`. and each of which has an increasing average probability of *conversion*, respectively. The simulated data also contains four `attribute` columns, named `attr_*`.

```python
from abra.utils import generate_fake_observations

# generate demo data
experiment_observations = generate_fake_observations(
    distribution='bernoulli',
    n_treatments=3,
    n_attributes=4,
    n_observations=120
)

experiment_observations.head()
"""
   id treatment attr_0 attr_1 attr_2 attr_3  metric
0   0         C    A0a    A1a    A2a    A3a     1.0
1   1         B    A0b    A1a    A2a    A3a     1.0
2   2         C    A0c    A1a    A2a    A3a     1.0
3   3         C    A0c    A1a    A2a    A3a     0.0
4   4         A    A0b    A1a    A2a    A3a     1.0
"""
```

### Running an AB test in ✨ABracadabra✨ is as easy as ✨123✨:

The three key components of running an AB test are:

- **The `Experiment`**, which references the observations recorded during experiment (described above) and any optional metadata associated with the experiment.
- **The `HypothesisTest`**, which defines the hypothesis and statistical inference method applied to the experiment data.
- **The `HypothesisTestResults`**, which is the statistical artifact that results from running a `HypothesisTest` against an `Experiment`'s observations. The `HypothesisTestResults` are used to summarize, visualize, and interpret the inference results and make decisions based on these results.

Thus running an hypothesiss test in ✨ABracadabra✨ follows the basic 123 pattern:

1. Initialize your `Experiment` with observations and (optionally) any associated metadata.
2. Define your `HypothesisTest`. This requires defining the `hypothesis` and a relevant `inference_method`, which will depend on the support of your observations.
3. Run the test against your experiment and interpret the resulting `HypothesisTestResults`

We now demonstrate how to run and analyze a hypothesis test on the artificial observations data generated above. Since this simulated experiment focuses on a binary `metric` we'll want our `HypothesisTest` to use an `inference_method` that supports binary variables. The `"proportions_delta"` inference method, which tests for a significant difference in average probability between two different samples of probabilities is a valid test for our needs. Here our probabilities equal either `0` or `1`, but the sample averages will likely be equal to some intermediate value. This is analogous to AB tests that aim to compare conversion rates between a control and a variation group.

In addition to the `inference_method`, we also want to establish the `hypothesis` we want to test. In other words, if we find a significant difference in conversion rates, do we expect one group to be larger or smaller than the other. In this test we'll test that the `variation` group `"C"`has a `"larger"` average conversion rate than the `control` group `"A"`.

Below we show how to run such a test in ✨ABracadabra✨.

```python
# Running an AB Test is as easy as 1, 2, 3
from abra import Experiment, HypothesisTest

# 1. Initialize the `Experiment`
# We (optionally) name the experiment "Demo"
exp = Experiment(data=experiment_observations, name='Demo')

# 2. Define the `HypothesisTest`
# Here, we test that the variation "C" is "larger" than the control "A",
# based on the values of the "metric" column, using a Frequentist z-test,
# as parameterized by `inference_method="proportions_delta"`
ab_test = HypothesisTest(
    metric='metric',
    treatment='treatment',
    control='A', variation='C',
    inference_method='proportions_delta',
    hypothesis='larger'
)

# 3. Run and interpret the `HypothesisTestResults`
# Here, we run our HypothesisTest with an assumed
# Type I error rate of alpha=0.05
ab_test_results = exp.run_test(ab_test, alpha=.05)
assert ab_test_results.accept_hypothesis

# Display results
ab_test_results.display()
"""
Observations Summary:
+----------------+------------------+------------------+
| Treatment      | A                | C                |
+----------------+------------------+------------------+
| Metric         | metric           | metric           |
| Observations   | 35               | 44               |
| Mean           | 0.4286           | 0.6136           |
| Standard Error | (0.2646, 0.5925) | (0.4698, 0.7575) |
| Variance       | 0.2449           | 0.2371           |
+----------------+------------------+------------------+

Test Results:
+---------------------------+---------------------+
| ProportionsDelta          | 0.1851              |
| ProportionsDelta CI       | (-0.0000, inf)      |
| CI %-tiles                | (0.0500, inf)       |
| ProportionsDelta-relative | 43.18 %             |
| CI-relative               | (-0.00, inf) %      |
| Effect Size               | 0.3773              |
| alpha                     | 0.0500              |
| Power                     | 0.5084              |
| Inference Method          | 'proportions_delta' |
| Test Statistic ('z')      | 1.91                |
| p-value                   | 0.0280              |
| Degrees of Freedom        | None                |
| Hypothesis                | 'C is larger'       |
| Accept Hypothesis         | True                |
| MC Correction             | None                |
| Warnings                  | None                |
+---------------------------+---------------------+
"""

# Visualize Frequentist Test results
ab_test_results.visualize(outfile="./images/proportions_delta_example.png")
```

![proportions_delta_inference_example](http://github.com/quizlet/abracadabra/blob/master/images/proportions_delta_example.png "proportions_delta Inference Example")

We see that the Hypothesis test declares that the variation `'C is larger'` (than the control `"A"`) showing a 43% relative increase in conversion rate, and a moderate effect size of 0.38. This results in a p-value of 0.028, which is lower than the prescribed $\alpha=0.05$.

## Bayesian AB Tests

Running Bayesian AB Tests is just as easy as running a Frequentist test, simply change the `inference_method` of the `HypothesisTest`. Here we run Bayesian hypothesis test that is analogous to `"proportions_delta"` used above for conversion rates. The Bayesian test is based on the [Beta-Binomial model](https://en.wikipedia.org/wiki/Beta-binomial_distribution), and thus called with the argument `inference_method="beta_binomial"`.

```python
# Copy the parameters of the original HypothesisTest,
# but update the `inference_method`
bayesian_ab_test = ab_test.copy(inference_method='beta_binomial')
bayesian_ab_test_results = exp.run_test(bayesian_ab_test)
assert bayesian_ab_test_results.accept_hypothesis

# Display results
bayesian_ab_test_results.display()
"""
Observations Summary:
+----------------+------------------+------------------+
| Treatment      | A                | C                |
+----------------+------------------+------------------+
| Metric         | metric           | metric           |
| Observations   | 35               | 44               |
| Mean           | 0.4286           | 0.6136           |
| Standard Error | (0.2646, 0.5925) | (0.4698, 0.7575) |
| Variance       | 0.2449           | 0.2371           |
+----------------+------------------+------------------+

Test Results:
+------------------------+-------------------------------+
| Delta                  | 0.1745                        |
| HDI                    | (-0.0276, 0.3818)             |
| HDI %-tiles            | (0.0500, 0.9500)              |
| Delta-relative         | 45.46 %                       |
| HDI-relative           | (-13.03, 111.61) %            |
| Effect Size            | 0.3631                        |
| alpha                  | 0.0500                        |
| Credible Mass          | 0.9500                        |
| p(C > A)               | 0.9500                        |
| Inference Method       | 'beta_binomial'               |
| Model Hyperarameters   | {'alpha_': 1.0, 'beta_': 1.0} |
| Inference Method       | 'sample'                      |
| Hypothesis             | 'C is larger'                 |
| Accept Hypothesis      | True                          |
| Warnings               | None                          |
+------------------------+-------------------------------+
"""

# Visualize Bayesian AB test results, including samples from the model
bayesian_ab_test_results.visualize(outfile="./images/beta_binomial_example.png")
```
![`beta_binomial_inference_example](https://github.com/quizlet/abracadabra/blob/master/images/beta_binomial_example.png "beta_binomial Inference Example")

Above we see that the Bayesian hypothesis test provides similar results to the Frequentist test, indicating a 45% relative lift in conversion rate when comparing `"C"` to `"A"`. Rather than providing p-values that are used to accept or reject a Null hypothesis, the Bayesian tests provides directly-interpretable probability estimates `p(C > A) = 0.95`, here indicating that there is 95% chance that the `variation` `"C"` is larger than the `control` `"A"`.

## [Additional Documentation and Tutorials](https://github.com/quizlet/abracadabra/blob/master/docs)

## CHANGELOG
- 2020-06-11: Initial release--`version=0.0.0`

## TODO / In Works
- Hypothesis Test `inference_method`s
    + Non-parametric Tests
        * Wilcox Rank-sum / Man-Whitney U
    + $\Chi^2$ Test for proportions