import pytest
import baby_lm


def test_sum_as_string():
    assert baby_lm.sum_as_string(1, 1) == "2"
