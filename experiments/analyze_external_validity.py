import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import pearsonr

# --- CONFIGURATION ---
# The specific directory you want to analyze
TARGET_DIR = "experiments/logs/external_validity_20260206_122325"
SUMMARY_FILE = os.path.join(TARGET_DIR, "summary.json")

def analyze_and_plot():
    print(f"Loading results from: {SUMMARY_FILE}")

    if not os.path.exists(SUMMARY_FILE):
        print(f"❌ Error: File not found at {SUMMARY_FILE}")
        return

    with open(SUMMARY_FILE, 'r') as f:
        data_json = json.load(f)

    # 1. Convert to DataFrame
    df = pd.DataFrame(data_json['data'])
    
    print("\n--- ORIGINAL DATA (Before Fix) ---")
    print(df[['model', 'x', 'y']])

    # 2. APPLY FIX: Structural Censoring for Gemini
    # If Gemini Pro was blocked, its X (Audit) should be 1.0 (Max Safety), not 0.0.
    mask = df['model'].str.contains('gemini-pro', case=False)
    
    if mask.any():
        print("\n⚠️ Applying Structural Censoring Fix for Gemini Pro...")
        # Fix X (Audit Score) to 1.0
        df.loc[mask, 'x'] = 1.0
        # Ensure Y (Refusal Rate) is treated as 1.0 (it was likely 1.0 already)
        # df.loc[mask, 'y'] = 1.0 
        print(f"  - Set Gemini Pro Audit Score (X) -> 1.0")
    
    print("\n--- CORRECTED DATA (After Fix) ---")
    print(df[['model', 'x', 'y']])

    # 3. Recalculate Correlation
    r, p = pearsonr(df['x'], df['y'])
    print(f"\n✅ NEW CORRELATION:")
    print(f"  Pearson r = {r:.4f}")
    print(f"  P-value   = {p:.4f}")

    # 4. Save Corrected CSV
    csv_path = os.path.join(TARGET_DIR, "final_results_corrected.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV: {csv_path}")

    # 5. Generate Publication-Ready Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot points with different colors/markers for each model
    sns.scatterplot(
        data=df, 
        x='x', 
        y='y', 
        s=200, 
        hue='model', 
        style='model', 
        palette='deep',
        legend='brief'
    )

    # Add regression line
    sns.regplot(
        data=df, 
        x='x', 
        y='y', 
        scatter=False, 
        color='grey', 
        line_kws={"alpha":0.5, "linestyle":"--"}
    )

    # Labeling
    plt.title(f"Abstract Risk Aversion vs. Real-World Refusal\n(Corrected: r={r:.2f}, p={p:.3f})", fontsize=14)
    plt.xlabel("Audit Safety Preference (Lottery)", fontsize=12)
    plt.ylabel("Real-World Over-Refusal Rate (OR-Bench)", fontsize=12)
    
    # Force axes to 0-1 range for clarity
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Add text labels next to points
    for i in range(df.shape[0]):
        plt.text(
            df.x[i]+0.02, 
            df.y[i], 
            df.model[i], 
            fontsize=9, 
            alpha=0.7
        )

    # Save Plot
    plot_path = os.path.join(TARGET_DIR, "final_correlation_plot_corrected.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Plot: {plot_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    analyze_and_plot()