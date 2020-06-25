import logging
import os
from abra.utils import generate_fake_observations
from abra import config


def test_logger_level():
    assert config.logger.level == getattr(logging, (config.get('core', 'logging_level')))


def test_expand_env_var():
    assert 'blammo' == config.expand_env_var("blammo")


def test_render_config_template():
    template = "{ABRACADABRA_USER}"
    assert config.render_config_template(template) == config.ABRACADABRA_USER


def test_home():
    assert os.environ['ABRACADABRA_HOME'] == config.ABRACADABRA_HOME


def test_user():
    assert os.environ['ABRACADABRA_USER'] == config.ABRACADABRA_USER


def test_config_file():
    assert os.path.isfile(config.ABRACADABRA_CONFIG)


def test_coerce_value():
    assert config.coerce_value('true') is True
    assert config.coerce_value('false') is False
    assert isinstance(config.coerce_value('1.0'), float)
    assert isinstance(config.coerce_value('1'), int)
    assert isinstance(config.coerce_value('a,b'), list)


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
