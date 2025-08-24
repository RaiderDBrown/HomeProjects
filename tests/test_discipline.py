import os
import sys

import pandas as pd
from scipy import stats
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from schoolboard import discipline

def test_t_test_matches_ttest_1samp():
    data = pd.Series([1.0, 2.0, 3.0], name="demo")
    expected = stats.ttest_1samp(data, 0).pvalue
    result = discipline.t_test(data)
    assert result == pytest.approx(expected)
