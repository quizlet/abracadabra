import logging
from abra.utils import generate_fake_observations
from abra import config


def test_logger_level():
    assert config.logger.level == getattr(logging, (config.get('core', 'logging_level')))


def test_get_set():
    config.set('core', 'test_value', 'test')
    assert config.get('core', 'test_value') == 'test'
    assert config.CONFIG.get('core', 'test_value') == 'test'


def test_search_config():
    df = generate_fake_observations(n_observations=1)

    # test against default config template
    assert 'treatment' in config.search_config(df, 'experiment', 'treatment')
    assert 'metric' in config.search_config(df, 'experiment', 'measures')
    assert 'attr_0' in config.search_config(df, 'experiment', 'attributes')
