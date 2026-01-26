"""
Tests for uncertainty profiling.
"""

import pytest
import numpy as np
import gpdiagnostics

class TestUncertaintyProfiler:
    def test_profiler_initialization(self, simple_gp):
        """Test profiler creates correctly."""
        profiler = gpdiagnostics.UncertaintyProfiler(simple_gp)
        assert profiler.model is simple_gp
    
    def test_predict_with_uncertainty(self, simple_gp, X_test):
        """Test prediction method."""
        profiler = gpdiagnostics.UncertaintyProfiler(simple_gp)
        mean, var = profiler.predict_with_uncertainty(X_test)
        
        assert mean.shape == (X_test.shape[0], 1)
        assert var.shape == (X_test.shape[0], 1)
        assert np.all(var >= 0)  # Variance must be non-negative
    
    def test_compute_diagnostics(self, simple_gp, X_test):
        """Test diagnostic computation."""
        profiler = gpdiagnostics.UncertaintyProfiler(simple_gp)
        diagnostics = profiler.compute_diagnostics(X_test)
        
        required_keys = [
            "mean_uncertainty", "max_uncertainty", "uncertainty_std",
            "total_uncertainty", "high_uncertainty_ratio"
        ]
        for key in required_keys:
            assert key in diagnostics
            assert isinstance(diagnostics[key], float)
    
    def test_identify_uncertainty_regions(self, simple_gp, X_test):
        """Test region identification."""
        profiler = gpdiagnostics.UncertaintyProfiler(simple_gp)
        regions = profiler.identify_uncertainty_regions(X_test, threshold_percentile=90)
        
        assert "high_uncertainty_points" in regions
        assert "low_uncertainty_points" in regions
        assert "threshold" in regions
        assert len(regions["high_uncertainty_points"]) > 0
