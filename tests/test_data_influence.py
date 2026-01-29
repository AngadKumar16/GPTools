"""
Tests for data influence analysis.
"""

import GPy
import numpy as np
import pytest

import gptools


class TestDataInfluenceMap:
    def test_influence_map_initialization(self, simple_gp):
        """Test influence map creation."""
        influence_map = gptools.DataInfluenceMap(simple_gp)
        assert influence_map.model is simple_gp

    def test_compute_influence_scores(self, simple_gp):
        """Test influence score computation."""
        influence_map = gptools.DataInfluenceMap(simple_gp)
        scores = influence_map.compute_influence_scores(simple_gp.X)

        assert len(scores) == simple_gp.X.shape[0]
        assert np.all(scores > 0)
        assert not np.any(np.isnan(scores))

    def test_get_influence_report(self, simple_gp):
        """Test comprehensive influence report."""
        influence_map = gptools.DataInfluenceMap(simple_gp)
        report = influence_map.get_influence_report(simple_gp.X, simple_gp.Y.flatten())

        assert "influence_scores" in report
        assert "most_influential_point" in report
        assert "least_influential_point" in report
        assert (
            report["most_influential_point"]["score"]
            > report["least_influential_point"]["score"]
        )

    def test_loo_variance_increase(self, simple_gp):
        """Test LOO variance computation."""
        influence_map = gptools.DataInfluenceMap(simple_gp)
        var_inc, errors = influence_map.compute_loo_variance_increase(
            simple_gp.X, simple_gp.Y.flatten()
        )

        assert len(var_inc) == simple_gp.X.shape[0]
        assert len(errors) == simple_gp.X.shape[0]
