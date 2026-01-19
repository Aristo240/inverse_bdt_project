import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Groups
SMALL_MODELS = ["mistral_7b", "llama3_8b", "qwen2.5_7b", "deepseek_7b", "gemma2_9b"]
FRONTIER_MODELS = ["llama3_70b", "gemma2_27b", "gpt-4o"]

PRETTY_NAMES = {
    "mistral_7b": "Mistral 7B", "llama3_8b": "Llama-3 8B", "gemma2_9b": "Gemma-2 9B",
    "qwen2.5_7b": "Qwen 2.5", "deepseek_7b": "DeepSeek",
    "gemma2_27b": "Gemma-2 27B", "llama3_70b": "Llama-3 70B", "gpt-4o": "GPT-4o",
}

def load_radar_data():
    records = []
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} not found.")
        return pd.DataFrame()

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                model_id = data.get("model", filename.replace(".json", ""))
                
                if "gemini" in model_id.lower():
                    continue

                exps = data.get("experiments", {})
                risk_intolerance = exps.get("microrisk", {}).get("safe_pct", 0)
                incorruptibility = exps.get("godfather", {}).get("safe_pct", 0)
                harm_sensitivity = exps.get("sensitivity", {}).get("safe_pct", 0)
                censor_rate = exps.get("microrisk", {}).get("censor_rate", 0)
                stability = 1.0 - censor_rate
                
                plot_group = "Other"
                if any(m in model_id for m in SMALL_MODELS):
                    plot_group = "Small (Avg)"
                elif any(m in model_id for m in FRONTIER_MODELS):
                    plot_group = "Frontier"

                records.append({
                    "Model ID": model_id,
                    "Name": PRETTY_NAMES.get(model_id, model_id),
                    "Plot Group": plot_group,
                    "Risk Intolerance": risk_intolerance,
                    "Incorruptibility": incorruptibility,
                    "Harm Sensitivity": harm_sensitivity,
                    "Stability": stability
                })
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                
    return pd.DataFrame(records)

def plot_radar(df):
    if df.empty:
        print("No data to plot.")
        return

    # Categories
    categories = ['Risk Intolerance', 'Incorruptibility', 'Harm Sensitivity', 'Stability']
    N = len(categories)
    
    # Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # --- FIX 1: Add Padding so labels don't cut off ---
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

    # --- FIX 2: Better Labels with Newlines ---
    labels = [
        "Risk Intolerance\n(Refuses Low Prob)", 
        "Incorruptibility\n(Refuses High Reward)", 
        "Harm Sensitivity\n(Refuses Low Harm)", 
        "Stability\n(Valid Responses)"
    ]
    plt.xticks(angles[:-1], labels, color='black', size=11, weight='bold')
    
    # --- FIX 3: Move Radial Labels out of the way ---
    ax.set_rlabel_position(45)  # Move the 0.25, 0.50 text to 45 degrees
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    # --- PLOTTING ---
    # 1. Small Models (Average)
    small_df = df[df["Plot Group"] == "Small (Avg)"]
    if not small_df.empty:
        mean_vals = small_df[categories].mean().tolist()
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, linewidth=2, linestyle='--', color="#9b59b6", label="Small Models (Avg)")
        ax.fill(angles, mean_vals, color="#9b59b6", alpha=0.15)

    # 2. Frontier Models (Individual)
    frontier_styles = {
        "gpt-4o":     {"color": "#27ae60", "style": "-", "width": 3},  # Green
        "llama3_70b": {"color": "#e67e22", "style": "-", "width": 3},  # Orange
        "gemma2_27b": {"color": "#2980b9", "style": ":", "width": 2},  # Blue
    }

    frontier_df = df[df["Plot Group"] == "Frontier"]
    for i, row in frontier_df.iterrows():
        style_dict = {"color": "gray", "style": "-", "width": 1}
        for k, v in frontier_styles.items():
            if k in row["Model ID"]:
                style_dict = v
                break
        
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=style_dict["width"], linestyle=style_dict["style"], 
                color=style_dict["color"], label=row["Name"])

    # Legend
    plt.title("Safety Phenotypes: The 'Fingerprint' of Alignment", size=16, weight='bold', y=1.1)
    # Move legend further out
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.10))
    
    out_path = os.path.join(OUTPUT_DIR, "safety_radar.jpg")
    plt.savefig(out_path, dpi=300)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    df = load_radar_data()
    plot_radar(df)