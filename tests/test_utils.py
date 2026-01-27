"""
Tests for utility functions.
"""

import GPy
import numpy as np
import pytest

import gpdiagnostics


class TestUtils:
    def test_get_lengthscale(self, simple_gp):
        """Test lengthscale extraction."""
        ls = gpdiagnostics.get_lengthscale(simple_gp)
        assert ls is not None
        assert ls > 0

    def test_get_noise_variance(self, simple_gp):
        """Test noise variance extraction."""
        nv = gpdiagnostics.get_noise_variance(simple_gp)
        assert nv is not None
        assert nv > 0

    def test_extract_kernel_params_flat(self, simple_gp):
        """Test parameter flattening."""
        params = gpdiagnostics.extract_kernel_params_flat(simple_gp)
        assert isinstance(params, dict)
        assert len(params) > 0
        assert all(isinstance(v, float) for v in params.values())

    def test_check_model_health(self, simple_gp):
        """Test model health check."""
        health = gpdiagnostics.check_model_health(simple_gp)
        assert isinstance(health, dict)
        assert "healthy" in health
        assert isinstance(health["healthy"], bool)

    def test_check_model_health_invalid(self):
        """Test health check on problematic model."""
        # Create model with NaN parameters
        X = np.random.rand(10, 1)
        y = np.random.rand(10)

        kernel = GPy.kern.RBF(1)
        model = GPy.models.GPRegression(X, y[:, None], kernel)
        # Force NaN in parameters
        model.kern.lengthscale = np.nan

        health = gpdiagnostics.check_model_health(model)
        assert health["healthy"] is False
