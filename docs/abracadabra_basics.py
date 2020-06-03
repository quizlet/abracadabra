# generate some fake binary data
from abra.utils import generate_fake_results
from abra import Experiment, HypothesisTest


def print_section_header(msg):
    brk = "-" * len(msg)
    print(brk)
    print(msg)
    print(brk)


print_section_header("Input Data:")

from abra.utils import generate_fake_results

# generate some fake binary trial data
data = generate_fake_results(
    distribution='bernoulli',  # binary data
    n_treatments=3,
    n_attributes=4,
    n_observations=1000
)
print(data.head())


print_section_header("API Usage:")
# initialize the Experiment
exp = Experiment(data=data, name='test')

# set up an A/B test
ab_test = HypothesisTest(
    metric='metric',
    treatment='treatment',
    control='A', variation='B',
    hypothesis='larger',
    method='proportions_delta'
)

# run the test
ab_test_results = exp.run_test(ab_test, alpha=.05)
assert ab_test_results.accept_hypothesis