"""
Utility functions for common GP diagnostics tasks.
"""

import numpy as np
import GPy
from typing import Optional, Union, Dict

def get_lengthscale(model: GPy.models.GPRegression) -> Optional[Union[float, np.ndarray]]:
    """
    Safely extract lengthscale from any kernel type.
    
    Args:
        model: Trained GPy model
        
    Returns:
        Lengthscale value(s) or None if not found
    """
    try:
        if hasattr(model.kern, 'lengthscale'):
            ls = model.kern.lengthscale
            return ls if hasattr(ls, '__iter__') else float(ls)
    except Exception:
        return None
    return None

def get_noise_variance(model: GPy.models.GPRegression) -> Optional[float]:
    """
    Safely extract observation noise variance.
    
    Args:
        model: Trained GPy model
        
    Returns:
        Noise variance or None if not found
    """
    try:
        return float(model.Gaussian_noise.variance)
    except Exception:
        return None

def extract_kernel_params_flat(model: GPy.models.GPRegression) -> Dict[str, float]:
    """
    Extract all kernel parameters as flat key-value pairs.
    
    Handles vector parameters by flattening with dimension indices.
    
    Args:
        model: Trained GPy model
        
    Returns:
        Flat dictionary of parameters
    """
    params = {}
    for param in model.parameters:
        val = param.param_array
        if hasattr(val, '__iter__') and len(val) > 1:
            for i, v in enumerate(val):
                params[f"{param.name}_{i}"] = float(v)
        elif hasattr(val, '__iter__'):
            params[param.name] = float(val[0])
        else:
            params[param.name] = float(val)
    return params

def check_model_health(model: GPy.models.GPRegression) -> Dict[str, bool]:
    """
    Quick health check for GP model.
    
    Args:
        model: Trained GPy model
        
    Returns:
        Dictionary of health indicators
    """
    try:
        # Check for NaN parameters
        nan_params = any(np.any(np.isnan(p.param_array)) for p in model.parameters)
        
        # Check noise variance is reasonable
        noise_ok = True
        noise = get_noise_variance(model)
        if noise is not None:
            noise_ok = noise > 0 and noise < 1e6
        
        # Check kernel variance
        kern_var_ok = True
        if hasattr(model.kern, 'variance'):
            kv = float(model.kern.variance)
            kern_var_ok = kv > 0 and kv < 1e6
        
        return {
            "valid_parameters": not nan_params,
            "reasonable_noise": bool(noise_ok),
            "reasonable_variance": bool(kern_var_ok),
            "healthy": not nan_params and noise_ok and kern_var_ok
        }
    except:
        return {"healthy": False}
