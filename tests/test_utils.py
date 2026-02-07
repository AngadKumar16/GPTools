"""
Tests for utility functions.
"""

import GPy
import numpy as np
import pytest

import gpclarity


class TestUtils:
    def test_get_lengthscale(self, simple_gp):
        """Test lengthscale extraction."""
        ls = gpclarity.get_lengthscale(simple_gp)
        assert ls is not None
        assert ls > 0

    def test_get_noise_variance(self, simple_gp):
        """Test noise variance extraction."""
        nv = gpclarity.get_noise_variance(simple_gp)
        assert nv is not None
        assert nv > 0

    def test_extract_kernel_params_flat(self, simple_gp):
        """Test parameter flattening."""
        params = gpclarity.extract_kernel_params_flat(simple_gp)
        assert isinstance(params, dict)
        assert len(params) > 0
        assert all(isinstance(v, float) for v in params.values())

    def test_check_model_health(self, simple_gp):
        """Test model health check."""
        health = gpclarity.check_model_health(simple_gp)
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

        health = gpclarity.check_model_health(model)
        assert health["healthy"] is False

    def test_check_model_health(self):
        """Test model health checking."""
        # Healthy model
        health = gpclarity.check_model_health(self.model)
        self.assertTrue(health['is_healthy'])
        self.assertEqual(len(health['issues']), 0)
        self.assertIn('log_likelihood', health)
        
        # Model without predict
        bad_model = object()
        health = gpclarity.check_model_health(bad_model)
        self.assertFalse(health['is_healthy'])
        self.assertIn('Model missing predict() method', health['issues'])
    def test_get_lengthscale(self):
        """Test lengthscale extraction."""
        ls = gpclarity.get_lengthscale(self.model)
        self.assertIsInstance(ls, float)
        self.assertGreater(ls, 0)

    def test_get_noise_variance(self):
        """Test noise variance extraction."""
        noise = gpclarity.get_noise_variance(self.model)
        self.assertIsInstance(noise, float)
        self.assertGreaterEqual(noise, 0)

    def test_extract_kernel_params_flat(self):
        """Test flat parameter extraction."""
        params = gpclarity.extract_kernel_params_flat(self.model)
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
    
