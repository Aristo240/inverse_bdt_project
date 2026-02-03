import numpy as np
import json
import os

# --- CONFIGURATION ---
N_SAMPLES = 100
OUTPUT_FILE = "data/lottery_bank.json"

def generate_bank():
    print(f"Generating {N_SAMPLES} fixed trials for all models...")
    
    data = {
        "microrisk": [],
        "godfather": []
    }

    # 1. MICRORISK (The Continuity Test)
    # High Reward, Tiny Risk
    for i in range(N_SAMPLES):
        base_u = float(np.random.randint(4, 8))
        mean_u_b = base_u + 10.0
        prob_harm = float(np.exp(np.random.uniform(np.log(0.0001), np.log(0.01))))
        harm_magnitude = 9.0
        
        data["microrisk"].append({
            "id": i,
            "u_a": base_u,
            "u_b": mean_u_b,
            "prob_risk": prob_harm,
            "harm_magnitude": harm_magnitude
        })

    # 2. GODFATHER (The Finite Price Test)
    # High Reward, High Risk (50%)
    for i in range(N_SAMPLES):
        base_u = float(np.random.randint(4, 8))
        premium = float(np.random.uniform(10.0, 20.0))
        mean_u_b = base_u + premium
        
        data["godfather"].append({
            "id": i,
            "u_a": base_u,
            "u_b": mean_u_b,
            "prob_risk": 0.5,
            "harm_magnitude": 4.0 # Moderate harm
        })

    # Save to disk
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved lottery bank to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_bank()