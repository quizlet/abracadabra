import pytest
import numpy as np
from abra import utils


def test_dict_to_object():
    d = utils.dict_to_object({"a": "b"})
    assert d.a == 'b'


def test_ensure_dataframe(proportions_data_small):
    with pytest.raises(ValueError):
        utils.ensure_dataframe(None)
    assert utils.ensure_dataframe(proportions_data_small).equals(proportions_data_small)

    class DataObj:
        data = proportions_data_small

    assert utils.ensure_dataframe(DataObj(), 'data').equals(proportions_data_small)


def test_set_backend():
    assert utils.set_backend() in ('pdf', 'agg')

def test_safe_isnan():
    assert utils.safe_isnan(None) is False
    assert utils.safe_isnan(np.inf) == False
    assert utils.safe_isnan(np.nan) == True
