import pytest
from abra.utils import generate_fake_observations

def pytest_addoption(parser):
    parser.addoption(
        "--run_stan_tests", action="store_true", default=False,
        help="Run Stan tests which are skipped by default. "
        "The first time these tests are run requires compiling C++ code, "
        "which is cached locally and used afterwards"
    )


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# https://docs.pytest.org/en/latest/mark.html
def pytest_collection_modifyitems(config, items):
    """
    Notes
    -----
    When adding a new marker, make sure to also register it in the `markers` section of
    pytest.ini.
    """
    if not config.getoption("--run_stan_tests"):
        skip_slow_tests = pytest.mark.skip(
            reason="Skipping Stan tests - use --run_stan_tests to run"
        )
        for item in items:
            if "stan_test" in item.keywords:
                item.add_marker(skip_slow_tests)

# Put any global fixtures here (https://docs.pytest.org/en/latest/fixture.html)

@pytest.fixture()
def proportions_data_large():
    return generate_fake_observations(
        distribution='bernoulli',
        n_treatments=3,
        n_attributes=4,
        n_observations=10000
    )


@pytest.fixture()
def proportions_data_small():
    return generate_fake_observations(
        distribution='bernoulli',
        n_treatments=6,
        n_observations=6 * 50
    )


@pytest.fixture()
def means_data():
    return generate_fake_observations(
        distribution='gaussian',
        n_treatments=6,
        n_observations=6 * 50
    )


@pytest.fixture()
def counts_data():
    return generate_fake_observations(
        distribution='poisson',
        n_treatments=3,
        n_observations=3 * 100
    )
