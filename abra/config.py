# -*- coding: utf-8 -*-
import os
import logging
from configparser import ConfigParser
import getpass

# TODO: SET UP LOGGER FROM CONFIG

TEMPLATE_BEGIN_PATTERN = (
    '# ----------------------- BEGIN TEMPLATE -----------------------')


def in_test_mode():
    return os.environ.get('ABRACADABRA_TEST', 'false').lower() == 'true'


def expand_env_var(env_var):
    """
    Expands (potentially nested) env vars by repeatedly applying
    `expandvars` and `expanduser` until interpolation stops having
    any effect.
    """
    if not env_var:
        return env_var
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated


# load template, for constructing config files
with open(os.path.join(os.path.dirname(__file__), 'config_template.cfg')) as f:
    DEFAULT_CONFIG = f.read()


def render_config_template(template):
    """
    Generates a configuration from the provided template + variables defined in
    current scope
    :param template: a config content templated with {{variables}}
    """
    all_vars = {k: v for d in [globals(), locals()] for k, v in d.items()}
    return template.format(**all_vars)


# ConfigParser is "old-style" class which is a classobj, not a type.
# We thus use multiple inheritance with object to fix
class AbracadabraConfigParser(ConfigParser, object):
    """
    Custom config parser, with some validations
    """
    def __init__(self, *args, **kwargs):
        super(AbracadabraConfigParser, self).__init__(*args, **kwargs)
        self.is_validated = False

    def _validate(self):
        self.is_validated = True

    def read(self, filenames):
        ConfigParser.read(self, filenames)
        self._validate()


def mk_dir(dirname):
    if not os.path.isdir(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            raise Exception('Could not create directory {}:\n{}'.format(dirname, e))


# Home directory and configuration locations.
# We default to ~/abracadabra and ~/abracadabra/abracadabra.cfg if not provided
if 'ABRACADABRA_HOME' not in os.environ:
    ABRACADABRA_HOME = expand_env_var('~/abracadabra')
    os.environ['ABRACADABRA_HOME'] = ABRACADABRA_HOME
else:
    ABRACADABRA_HOME = expand_env_var(os.environ['ABRACADABRA_HOME'])

mk_dir(ABRACADABRA_HOME)

if 'ABRACADABRA_CONFIG' not in os.environ:
    if os.path.isfile(expand_env_var('~/abracadabra.cfg')):
        ABRACADABRA_CONFIG = expand_env_var('~/abracadabra.cfg')
    else:
        ABRACADABRA_CONFIG = ABRACADABRA_HOME + '/abracadabra.cfg'
else:
    ABRACADABRA_CONFIG = expand_env_var(os.environ['ABRACADABRA_CONFIG'])


if 'ABRACADABRA_USER' not in os.environ:
    ABRACADABRA_USER = getpass.getuser()
    os.environ['ABRACADABRA_USER'] = ABRACADABRA_USER
else:
    ABRACADABRA_USER = os.environ['ABRACADABRA_USER']

# Write the config file, if needed
if not os.path.isfile(ABRACADABRA_CONFIG):
    logging.info(f'Creating new Abracadabra config file in: {ABRACADABRA_CONFIG}')
    with open(ABRACADABRA_CONFIG, 'w') as f:
        cfg = render_config_template(DEFAULT_CONFIG)
        f.write(cfg.split(TEMPLATE_BEGIN_PATTERN)[-1].strip())


def coerce_value(val):
    """
    Coerce config variables to proper types
    """
    def isnumeric(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    if isnumeric(val):
        try:
            return int(val)
        except ValueError:
            return float(val)

    lower_val = str(val.lower())
    if lower_val in ('true', 'false'):
        if 'f' in lower_val:
            return False
        else:
            return True

    if ',' in val:
        return [coerce_value(v.strip()) for v in val.split(',')]
    return val


CONFIG = AbracadabraConfigParser()
CONFIG.read(ABRACADABRA_CONFIG)


# Give the entire module get/set methods
def get(section, key, **kwargs):
    """
    Retrieve typed variables from config.

    Example
    -------
    from abra import config
    # print the currently-configure ABRACADABRA_HOME directory
    print(config.get('core', 'abracadabra_home'))
    """
    return coerce_value(CONFIG.get(section, key, **kwargs))


def set(section, option, value, update=False):
    CONFIG.set(section, option, value)


def search_config(df, section, key):
    """
    Search a dataframe `df` for parameters defined in the global configuration.

    Parameters
    ----------
    df: dataframe
        raw data to search
    param_name: str
        type of parameter to search ('entities', 'metrics', or 'attributes')
    """
    available = get(section, key)
    available = [available] if not isinstance(available, list) else available
    columns = df.columns
    return [c for c in columns if c in available]



DEFAULT_ALPHA = get('constants', 'default_alpha')
MIN_OBS_FOR_Z = get('constants', 'min_obs_for_z')

STAN_MODEL_CACHE = get('stan', 'model_cache')
DEFAULT_BAYESIAN_INFERENCE_METHOD = get('stan', 'default_bayesian_inference_method')
