"""
Tests for hyperparameter tracking.
"""

import GPy
import numpy as np
import pytest

import gpdiagnostics


class TestHyperparameterTracker:
    def test_tracker_initialization(self, simple_gp):
        """Test tracker initialization."""
        tracker = gpdiagnostics.HyperparameterTracker(simple_gp)
        assert tracker.model is simple_gp
        assert tracker.history == {}

    def test_record_state(self, simple_gp):
        """Test state recording."""
        tracker = gpdiagnostics.HyperparameterTracker(simple_gp)
        tracker.record_state()

        assert len(tracker.history) > 0
        for param_name, values in tracker.history.items():
            assert len(values) == 1

    def test_wrapped_optimize(self, simple_gp):
        """Test optimization wrapping."""
        tracker = gpdiagnostics.HyperparameterTracker(simple_gp)
        history = tracker.wrapped_optimize(max_iters=5)

        assert len(history) > 0
        assert tracker.iteration_count == 5
        for param_name, values in history.items():
            assert len(values) == 5

    def test_convergence_report(self, simple_gp):
        """Test convergence analysis."""
        tracker = gpdiagnostics.HyperparameterTracker(simple_gp)
        tracker.wrapped_optimize(max_iters=10)
        report = tracker.get_convergence_report()

        assert isinstance(report, dict)
        if report:  # If optimization ran successfully
            for param, stats in report.items():
                assert "relative_change" in stats
                assert "converged" in stats
