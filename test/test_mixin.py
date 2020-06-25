import pytest
import numpy as np
import json
from abra import mixin

def test_init_repr():
    class TestRepr(mixin.InitRepr):
        __ATTRS__ = ['abra']
        abra = 'cadabra'

    ir = TestRepr()
    assert ir.__repr__() == "TestRepr(abra='cadabra')"

def test_jsonable():
    class TestJsonable(mixin.Jsonable):
        def __init__(self):
            self.abra = 'cadabra'

    ts = TestJsonable()
    assert 'abra' in ts.json
    assert ts.json['abra'] == 'cadabra'
    assert json.loads(ts.__repr__()) == ts.json

def test_dataframeable():
    class ValidDataFrameable(mixin.Dataframeable):
        def _json(self):
            return {'abra': ['cadabra', 'calamazam']}

    class InvalidDataframeable(mixin.Dataframeable):
        pass

    with pytest.raises(NotImplementedError):
        InvalidDataframeable().to_dataframe()

    vdf = ValidDataFrameable()
    assert vdf.json == {'abra': ['cadabra', 'calamazam']}
    df = vdf.to_dataframe()
    assert 'abra' in df.columns
    assert df.iloc[0].values == 'cadabra'
    assert df.iloc[1].values == 'calamazam'

def test_safe_isnan():
    assert mixin.safe_isnan(None) is False
    assert mixin.safe_isnan(np.inf) == False
    assert mixin.safe_isnan(np.nan) == True

