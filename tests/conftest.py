import pytest

from numba_utils import config


@pytest.fixture(autouse=True)
def _reset_global_config():
    yield
    config.reset()
