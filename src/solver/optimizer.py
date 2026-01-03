import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from typing import List, Tuple
from src.utils.structures import Lottery

def inverse_bdt_solver(
    lotteries_A: List[Lottery], 
    lotteries_B: List[Lottery], 
    choices: np.ndarray,
    l2_reg: float = 1e-4,
    force_linear: bool = False
):
    """
    Recovers (w, lambda).
    Returns: (params, success, raw_nll)
    """
    # 1. Pre-compute stats
    stats_A = [l.get_stats() for l in lotteries_A]
    stats_B = [l.get_stats() for l in lotteries_B]
    
    mu_A = np.array([s[0] for s in stats_A])
    var_A = np.array([s[1] for s in stats_A])
    mu_B = np.array([s[0] for s in stats_B])
    var_B = np.array([s[1] for s in stats_B])
    
    y = np.asarray(choices)

    # Helper to calculate NLL without regularization (for reporting)
    def calculate_raw_nll(params):
        if force_linear:
            w = params
            lam = 0.0
        else:
            w = params[:-1]
            lam = params[-1]
        
        w_sq = w**2
        v_A = np.dot(mu_A, w) - lam * np.dot(var_A, w_sq)
        v_B = np.dot(mu_B, w) - lam * np.dot(var_B, w_sq)
        
        # P(A)
        p_A = expit(v_A - v_B)
        p_A = np.clip(p_A, 1e-12, 1 - 1e-12) # Tighter clip for better NLL precision
        
        return -np.sum(y * np.log(p_A) + (1 - y) * np.log(1 - p_A))

    # The Objective Function (with Regularization for the Solver)
    def objective_with_reg(params):
        nll = calculate_raw_nll(params)
        
        # Regularization
        w_part = params if force_linear else params[:-1]
        reg = l2_reg * np.sum(w_part**2)
        
        if not force_linear:
            lam = params[-1]
            reg += l2_reg * (lam**2)
            
        return nll + reg

    # Constraints: Utility (idx 0) must be positive
    if force_linear:
        init_guess = np.array([1.0, -1.0, -1.0])
        bounds = [(0.0, None), (None, None), (None, None)]
    else:
        init_guess = np.array([1.0, -1.0, -1.0, 0.5])
        bounds = [(0.0, None), (None, None), (None, None), (0.0, None)]
    
    # Run Optimization
    result = minimize(
        objective_with_reg, 
        init_guess, 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    # Calculate the FINAL metric using only Raw NLL (Valid for comparison)
    final_raw_nll = calculate_raw_nll(result.x)
    
    return result.x, result.success, final_raw_nll