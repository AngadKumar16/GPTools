"""
Test fixtures and utilities for gpclarity.
"""

import GPy
import numpy as np
import pytest


@pytest.fixture
def simple_gp():
    """Simple 1D GP regression model."""
    np.random.seed(42)
    X = np.linspace(0, 10, 30).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.1 * np.random.randn(30)

    kernel = GPy.kern.RBF(1)
    model = GPy.models.GPRegression(X, y[:, None], kernel)
    model.optimize()
    return model


@pytest.fixture
def composite_gp():
    """GP with composite kernel."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.05 * np.random.randn(50)

    kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
    model = GPy.models.GPRegression(X, y[:, None], kernel)
    model.optimize()
    return model


@pytest.fixture
def X_test():
    """Test data points."""
    return np.linspace(-2, 12, 100).reshape(-1, 1)
