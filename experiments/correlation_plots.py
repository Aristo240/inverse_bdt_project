import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import pearsonr

def plot_final_results():
    # Your manual data from the text output
    data = {
        'model': ['mistral_7b', 'llama3_8b', 'gemma2_9b', 'qwen2.5_7b', 'gpt-4o', 'llama3_70b', 'gemini-pro'],
        'x': [0.957, 0.871, 0.9055, 0.9475, 0.9216, 0.977, 1.0],
        'y': [0.034, 0.030, 0.4088, 0.088, 0.098, 0.0346, 1.0],
        'family': ['Open Weights', 'Open Weights', 'Google', 'Open Weights', 'OpenAI', 'Open Weights', 'Google']
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stats
    r, p = pearsonr(df['x'], df['y'])
    print(f"Correlation: r={r:.4f}, p={p:.4f}")

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Create scatter plot with semantic coloring
    sns.scatterplot(
        data=df, 
        x='x', 
        y='y', 
        s=300, 
        hue='family', 
        style='family',
        palette='viridis' 
    )

    # Add regression line (optional, maybe leave out if it looks messy)
    sns.regplot(data=df, x='x', y='y', scatter=False, color='gray', line_kws={'alpha':0.3, 'linestyle':'--'})

    # Labels and Titles
    plt.title(f"Abstract Risk Aversion vs. Real-World Refusal\n(r={r:.2f}, p={p:.2f})", fontsize=16)
    plt.xlabel("Audit Safety Preference (Lottery)", fontsize=14)
    plt.ylabel("Real-World Over-Refusal Rate (OR-Bench)", fontsize=14)
    
    # Set limits to show the full range
    plt.xlim(0.8, 1.05)
    plt.ylim(-0.05, 1.05)

    # Annotate points
    for i in range(df.shape[0]):
        plt.text(
            df.x[i]+0.005, 
            df.y[i]+0.01, 
            df.model[i], 
            fontsize=11, 
            weight='bold'
        )

    # Save
    os.makedirs("experiments/paper_plots", exist_ok=True)
    plt.savefig("experiments/paper_plots/final_correlation.png", dpi=300, bbox_inches='tight')
    print("Plot saved to experiments/paper_plots/final_correlation.png")

if __name__ == "__main__":
    plot_final_results()