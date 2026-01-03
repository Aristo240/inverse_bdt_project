import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from typing import List, Tuple
from src.utils.structures import Lottery

def inverse_bdt_solver(
    lotteries_A: List[Lottery], 
    lotteries_B: List[Lottery], 
    choices: np.ndarray,
    l2_reg: float = 1e-4
):
    """
    Recovers (w, lambda) given observed choices.
    Returns: (params, success, debug_info)
    """
    n_samples = len(choices)
    print(f"Starting Inverse Optimization on {n_samples} samples...")
    
    # Sanity Check: Ensure data alignment
    if len(lotteries_A) != n_samples or len(lotteries_B) != n_samples:
        raise ValueError(f"Mismatch: {len(lotteries_A)} lotteries vs {n_samples} choices")

    # Sanity Check: Variance
    if len(np.unique(choices)) == 1:
        print("WARNING: All choices are identical. Solver may be unstable.")

    # Pre-compute statistics
    stats_A = [l.get_stats() for l in lotteries_A]
    stats_B = [l.get_stats() for l in lotteries_B]
    
    mu_A = np.array([s[0] for s in stats_A])
    var_A = np.array([s[1] for s in stats_A])
    mu_B = np.array([s[0] for s in stats_B])
    var_B = np.array([s[1] for s in stats_B])
    
    y = np.asarray(choices)

    def objective_nll(params):
        w = params[:-1]   # Weights
        lam = params[-1]  # Risk Penalty
        
        w_sq = w**2
        
        # BDT Utility: E[U] - lambda * Var(U)
        v_A = np.dot(mu_A, w) - lam * np.dot(var_A, w_sq)
        v_B = np.dot(mu_B, w) - lam * np.dot(var_B, w_sq)
        
        logits = v_A - v_B
        p_A = expit(logits)
        
        # Clip for stability
        p_A = np.clip(p_A, 1e-9, 1 - 1e-9)
        
        # NLL
        ll = -np.sum(y * np.log(p_A) + (1 - y) * np.log(1 - p_A))
        
        # L2 Regularization
        reg = l2_reg * (np.sum(w**2) + lam**2)
        
        return ll + reg

    init_guess = np.array([1.0, -1.0, -1.0, 0.5])
    bounds = [(None, None)] * 3 + [(0.0, None)]
    
    result = minimize(
        objective_nll, 
        init_guess, 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    return result.x, result.success, result