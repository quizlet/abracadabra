# ✨abracadabra✨

Makes AB testing analyses magically simple!

## Temporary setup for Analysts

This section outlines how to set up ✨abracadabra✨ in a separate environment for testing and going through demos before we publish the repo on PyPI. Note that once we get the go-ahead from T&S to open source the project, this entire process will be replaced with `pip install abracadabra`, but for now, please continue on...

#### Using the correct python version
✨abracadabra✨ is tested on python 3.7, so you'll need to make sure you're using the `3.7.0` runtime. First make sure that you have the `pyenv` package by running.

```bash
$ pyenv versions
system
2.7.15
2.7.16
* 3.7.0
```

This will list the versions of python on your machine. If you do not have `pyenv`, you'll need to install it via:

```bash
brew update
brew install pyenv
```

If version `3.7.0` is available, then great! Otherwise, you'll need to install it:

`pyenv install 3.7.0`

#### Set up a `virtualenv` for testing
Now (make and) go to a directory that you want to use to test ✨abracadabra✨. 

```bash
mkdir /tmp/test_abra && cd /tmp/test_abra
```

From there set the local python runtime.

```bash
pyenv local 3.7.0
```

Now make a virtual environment to install and test ✨abracadabra✨, then activate the `virtualenv`:

```bash
virtualenv abracadabra
source abracadabra/bin/activate
```

#### Install ✨abracadabra✨ and play
We can now install from Quizlet's github repo via.

```bash
pip install git+https://github.com/quizlet/abracadabra.git
```

This should have you up and running! Try it out by running the following example in fresh IPython shell. Once you're finished with that, try out the [`docs/abracadabra_basics.ipynb`](./docs/abracadabra_basics.ipynb) notebook, which gives a more in-depth look at all of ✨abracadabra✨'s components and features.

```bash
ipython
```



## Example Usage

```python
from abra.utils import generate_fake_observations
from abra import Experiment, HypothesisTest

# demo data
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

# Running an AB Test is as easy as 1, 2, 3

# 1. Initialize the `Experiment`
exp = Experiment(data=experiment_observations, name='demo')

# 2. Define the `HypothesisTest`
# Here, we test that the variation "C" is "larger" than the control "A",
# based on the values of the "metric" column, using a Frequentist z-test
ab_test = HypothesisTest(
    metric='metric',
    treatment='treatment',
    control='A', variation='C',
    hypothesis='larger',
    inference_method='proportions_delta'
)

# 3. Run and interpret the test results
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

![`proportions_delta` inference method](./images/proportions_delta_example.png)

## Bayesian AB Tests

Using Bayesian AB Tests are easy too, simply change the inference `method` of the `HypothesisTest`

```python
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
| p(variation > control) | 0.9500                        |
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
![`beta_binomial` inference method](./images/beta_binomial_example.png)

For additional examples see [`docs/abracadabra_basics.ipynb`](./docs/abracadabra_basics.ipynb)