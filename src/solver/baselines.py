import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from typing import List
from src.utils.structures import Lottery

# --- 1. PROSPECT THEORY (The "Human Bias" Model) ---

def weighting_function(p, gamma):
    """
    Standard Probability Weighting (Prelec-1).
    gamma = 1.0  -> Rational (p is just p)
    gamma < 1.0  -> Overweights small probabilities (Paranoia)
    """
    p = np.clip(p, 1e-9, 1.0) # Avoid log(0)
    return np.exp(-((-np.log(p)) ** gamma))

def solve_prospect_theory(
    lotteries_A: List[Lottery], 
    lotteries_B: List[Lottery], 
    choices: np.ndarray,
    l2_reg: float = 1e-4
):
    """
    Fits Multi-Attribute Prospect Theory.
    Params: w (Utility Weights) AND gamma (Probability Distortion).
    """
    y = np.array(choices)

    def get_pt_utility(lottery: Lottery, w: np.ndarray, gamma: float):
        total_util = 0.0
        # For every outcome in the lottery...
        for outcome, prob in zip(lottery.outcomes, lottery.probs):
            # 1. Calculate Linear Value (w * x) -> addressing the Multi-Attribute concern
            val = np.dot(outcome.features, w)
            
            # 2. Warp the probability using Gamma
            w_p = weighting_function(prob, gamma)
            
            # 3. Sum (Value * Distorted Prob)
            total_util += w_p * val
        return total_util

    def objective(params):
        # params = [w_util, w_harm, w_rep, gamma]
        w = params[:-1]
        gamma = params[-1]
        
        # Calculate scores for A and B
        u_A = np.array([get_pt_utility(l, w, gamma) for l in lotteries_A])
        u_B = np.array([get_pt_utility(l, w, gamma) for l in lotteries_B])
        
        # Probability of choosing A (Logistic Choice Rule)
        p_A = expit(u_A - u_B)
        p_A = np.clip(p_A, 1e-12, 1 - 1e-12)
        
        # Negative Log Likelihood
        nll = -np.sum(y * np.log(p_A) + (1 - y) * np.log(1 - p_A))
        
        # Regularization (Penalize w, keep gamma close to 1)
        reg = l2_reg * np.sum(w**2) + l2_reg * ((gamma - 1.0)**2)
        return nll + reg

    # Init: w=[1, -1, -1], gamma=1.0 (Rational Start)
    init_guess = np.array([1.0, -1.0, -1.0, 1.0])
    # Bounds: w is free, gamma must be positive (0.01 to 5.0)
    bounds = [(None, None), (None, None), (None, None), (0.01, 5.0)]
    
    result = minimize(objective, init_guess, method='L-BFGS-B', bounds=bounds)
    
    # Return Gamma (distortion) and the Score (NLL)
    return result.x[-1], result.fun

# --- 2. LEXICOGRAPHIC (The "Dumb Rule" Model) ---

def score_lexicographic(
    lotteries_A: List[Lottery], 
    lotteries_B: List[Lottery], 
    choices: np.ndarray
):
    """
    Tests the rule: "If Prob(Harm) > 0, Refuse (Choose A)."
    Returns Accuracy (0.0 to 1.0).
    """
    correct = 0
    total = len(choices)
    
    for i in range(total):
        l_B = lotteries_B[i]
        
        # Check if Option B has ANY risk (Harm > 0 with Prob > 0)
        has_risk = False
        for out, p in zip(l_B.outcomes, l_B.probs):
            # Feature index 1 is 'Safety Violation'
            if p > 0 and out.features[1] > 0:
                has_risk = True
                break
        
        # The Rule: If risk exists, choose A (1). Else choose B (0).
        # Note: Your generators always put the Risk in Option B.
        prediction = 1 if has_risk else 0
        
        if prediction == choices[i]:
            correct += 1
            
    return correct / total if total > 0 else 0.0