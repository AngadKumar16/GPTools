"""
Tests for model complexity scoring.
"""

import GPy
import numpy as np
import pytest

import gptools


class TestModelComplexity:
    def test_count_kernel_components_simple(self, simple_gp):
        """Test component counting for simple kernel."""
        count = gptools.count_kernel_components(simple_gp.kern)
        assert count == 1

    def test_count_kernel_components_composite(self, composite_gp):
        """Test component counting for composite kernel."""
        count = gptools.count_kernel_components(composite_gp.kern)
        assert count == 2

    def test_compute_roughness_score(self, simple_gp):
        """Test roughness score computation."""
        roughness = gptools.compute_roughness_score(simple_gp.kern)
        assert roughness > 0
        assert isinstance(roughness, float)

    def test_compute_noise_ratio(self, simple_gp):
        """Test noise ratio computation."""
        ratio = gptools.compute_noise_ratio(simple_gp)
        assert ratio > 0
        assert isinstance(ratio, float)

    def test_compute_complexity_score(self, simple_gp):
        """Test full complexity score."""
        X = simple_gp.X
        result = gptools.compute_complexity_score(simple_gp, X)

        assert "score" in result
        assert "interpretation" in result
        assert "components" in result
        assert isinstance(result["score"], float)
        assert result["score"] >= 0
