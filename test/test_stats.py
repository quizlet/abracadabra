import numpy as np
from abra.stats import Samples, MultipleComparisonCorrection


def test_multiple_comparison():
    p_values = np.arange(.001, .1, .01)
    mc = MultipleComparisonCorrection(p_values, method='b')

    assert mc.alpha_corrected < mc.alpha_orig
    assert mc.accept_hypothesis[0]
    assert not mc.accept_hypothesis[-1]

