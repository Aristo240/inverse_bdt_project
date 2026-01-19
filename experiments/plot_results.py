import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Full Model List
MODEL_ORDER = [
    "mistral_7b", "llama3_8b", "gemma2_9b", "qwen2.5_7b", "deepseek_7b", 
    "gemma2_27b", "llama3_70b", "gpt-4o", 
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro"
]

# 2. Approximate Sizes
MODEL_SIZES = {
    "mistral_7b": 7, 
    "llama3_8b": 8, 
    "gemma2_9b": 9, 
    "qwen2.5_7b": 7.5, # Slight tweak for visual separation on X-axis
    "deepseek_7b": 6.5, # Slight tweak for visual separation
    "gemma2_27b": 27, 
    "llama3_70b": 70, 
    "gpt-4o": 1000, 
    "gemini-2.5-pro": 1000, 
    "gemini-2.5-flash": 100, 
    "gemini-3-pro": 1500
}

# 3. Pretty Names
PRETTY_NAMES = {
    "mistral_7b": "Mistral 7B", 
    "llama3_8b": "Llama-3 8B", 
    "gemma2_9b": "Gemma-2 9B", 
    "qwen2.5_7b": "Qwen 2.5 7B", 
    "deepseek_7b": "DeepSeek 7B",
    "gemma2_27b": "Gemma-2 27B",
    "llama3_70b": "Llama-3 70B", 
    "gpt-4o": "GPT-4o",
    "gemini-2.5-pro": "Gemini 2.5 Pro", 
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3-pro": "Gemini 3 Pro"
}

# 4. Manual Offsets for Labels (Multiplier for X, Adder for Y)
# This fixes the overlapping labels by manually placing them
MANUAL_OFFSETS = {
    # Small cluster: Fan them out
    "mistral_7b":  (0.7,  0.08),  # Up and Left
    "llama3_8b":   (1.3,  0.08),  # Up and Right
    "gemma2_9b":   (1.3, -0.08),  # Down and Right
    "qwen2.5_7b":  (0.7, -0.08),  # Down and Left
    "deepseek_7b": (0.6,  0.00),  # Far Left (centered Y)
    
    # Large models
    "gpt-4o":      (1.2,  0.05),  # Slight push right
}

def load_data():
    records = []
    print(f"Looking for data in: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found at {DATA_DIR}")
        return pd.DataFrame()

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                model_name = data.get("model", filename.replace(".json", ""))
                matched_name = model_name
                for known in MODEL_ORDER:
                    if known in filename or known == model_name:
                        matched_name = known
                        break
                
                exp = data.get("experiments", {}).get("microrisk", {})
                censor_rate = exp.get("censor_rate", 0)
                safe_pct = exp.get("safe_pct", 0)
                gap = exp.get("gap", 0)

                if censor_rate > 0.9:
                    safe_pct = np.nan
                    gap = np.nan

                records.append({
                    "Model": matched_name,
                    "Label": PRETTY_NAMES.get(matched_name, matched_name),
                    "Size (B)": MODEL_SIZES.get(matched_name, 10),
                    "Safe %": safe_pct,
                    "BDT Gap": gap,
                    "Censor Rate": censor_rate,
                    "Type": "Google" if "gemini" in matched_name.lower() else "Standard"
                })
            except Exception as e:
                print(f"Skipping {filename}: {e}")
            
    return pd.DataFrame(records)

def plot_scaling_law(df):
    plt.figure(figsize=(14, 8)) 
    sns.set_style("whitegrid")
    
    trend_df = df[df["Type"] == "Standard"].sort_values("Size (B)")
    
    # 1. Plot Line & Points
    sns.lineplot(data=trend_df, x="Size (B)", y="Safe %", 
                 color="#2c3e50", linewidth=2, alpha=0.6, label="Risk Acceptance Trend")
    sns.scatterplot(data=trend_df, x="Size (B)", y="Safe %", 
                    s=150, color="#2980b9", edgecolor="black", zorder=10)
    
    # 2. Labeling Logic
    for i, row in enumerate(trend_df.itertuples()):
        model_key = row.Model
        
        # Default Zig-Zag
        x_mult = 1.0
        y_add = 0.04 if i % 2 == 0 else -0.06
        
        # Apply Manual Overrides if they exist
        if model_key in MANUAL_OFFSETS:
            x_mult, y_add = MANUAL_OFFSETS[model_key]

        plt.text(
            x=row._3 * x_mult,   # Size (B) scaled
            y=row._4 + y_add,    # Safe % shifted
            s=row.Label,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, 
            weight='bold', 
            color="#2c3e50",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    # 3. Google Models (Moved to Top Right)
    google_df = df[df["Type"] == "Google"]
    if not google_df.empty:
        plt.axvspan(90, 2000, color='#e74c3c', alpha=0.1, label="High Capability Zone")
        
        # MOVED UP: y=0.85 (Top Right Corner) to avoid GPT-4o overlap
        plt.text(
            1200, 0.85, 
            "GOOGLE MODELS\n(Gemini 2.5/3.0)\n\n100% CENSORED\n(Pre-Computation Veto)", 
            color='#c0392b', 
            fontsize=11, 
            fontweight='bold', 
            ha='center', 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c0392b", alpha=0.9)
        )

    plt.xscale("log")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0% (Risky)", "25%", "50%", "75%", "100% (Safe)"])
    plt.ylabel("Safety Refusal Rate", fontsize=13, labelpad=10)
    plt.xlabel("Model Parameters (Billions) - Log Scale", fontsize=13, labelpad=10)
    plt.title("The Behavioral Scaling Law: From Paralysis to Calculation", fontsize=16, weight='bold', pad=20)
    
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "scaling_law.jpg")
    plt.savefig(out_path, dpi=300)
    print(f"Generated {out_path}")

def plot_gap_analysis(df):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    plot_df = df.dropna(subset=["BDT Gap"]).sort_values("BDT Gap", ascending=True)
    
    if plot_df.empty:
        print("Skipping Gap Analysis - No valid data")
        return

    colors = []
    for x in plot_df["BDT Gap"]:
        if x < 1.0: colors.append("#27ae60")
        elif x < 3.0: colors.append("#f39c12")
        else: colors.append("#c0392b")

    bars = plt.barh(plot_df["Label"], plot_df["BDT Gap"], color=colors, height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.1, 
            bar.get_y() + bar.get_height()/2, 
            f"{width:.2f}", 
            ha='left', 
            va='center', 
            fontsize=10, 
            weight='bold'
        )

    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(2.0, color="gray", linestyle="--", alpha=0.7)
    plt.text(2.05, -1, "Non-Linearity Threshold", color="gray", fontsize=9, style='italic')

    plt.xlabel("BDT Gap (Deviation from Linear Utility)", fontsize=12, labelpad=10)
    plt.title("The Linearity Gap: Frontier Models Falsify RLHF Assumptions", fontsize=15, weight='bold', pad=20)
    
    plt.figtext(0.5, 0.02, 
                "Note: Gemini Models excluded due to 100% Censoring (No valid decisions to model)", 
                ha="center", fontsize=10, style='italic', color='#7f8c8d')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "bdt_gap.png")
    plt.savefig(out_path, dpi=300)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    print("--- Generating Plots for GitHub ---")
    df = load_data()
    
    if df.empty:
        print("❌ No valid .json files found! Check your ../data folder.")
    else:
        print(f"Loaded {len(df)} models.")
        plot_scaling_law(df)
        plot_gap_analysis(df)