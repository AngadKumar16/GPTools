"""
Tests for kernel summary and interpretation.
"""

import pytest
import gpdiagnostics

class TestKernelSummary:
    def test_summarize_kernel_simple(self, simple_gp):
        """Test kernel summary for simple RBF model."""
        summary = gpdiagnostics.summarize_kernel(simple_gp, verbose=False)
        
        assert "kernel_structure" in summary
        assert "components" in summary
        assert "composite" in summary
        assert not summary["composite"]
        assert len(summary["components"]) == 1
    
    def test_summarize_kernel_composite(self, composite_gp):
        """Test kernel summary for composite kernel."""
        summary = gpdiagnostics.summarize_kernel(composite_gp, verbose=False)
        
        assert summary["composite"]
        assert len(summary["components"]) == 2
    
    def test_interpret_lengthscale(self):
        """Test lengthscale interpretation logic."""
        assert "rapid" in gpdiagnostics.interpret_lengthscale(0.3).lower()
        assert "smooth" in gpdiagnostics.interpret_lengthscale(3.0).lower()
        assert "moderate" in gpdiagnostics.interpret_lengthscale(1.0).lower()
        
        # Test ARD lengthscales
        ard_ls = np.array([0.5, 1.5, 2.5])
        result = gpdiagnostics.interpret_lengthscale(ard_ls)
        assert "range" in result
    
    def test_interpret_variance(self):
        """Test variance interpretation logic."""
        low = gpdiagnostics.interpret_variance(0.001)
        assert "very low" in low.lower()
        
        high = gpdiagnostics.interpret_variance(15.0)
        assert "high" in high.lower()
        
        moderate = gpdiagnostics.interpret_variance(1.0)
        assert "moderate" in moderate.lower()
    
    def test_format_kernel_tree(self, composite_gp):
        """Test kernel tree formatting."""
        tree_str = gpdiagnostics.format_kernel_tree(composite_gp)
        assert "RBF" in tree_str
        assert "White" in tree_str
