import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# 1. The Data
data = {
    'model': ['Mistral 7B', 'Llama 3.1 8B', 'Llama 3.3 70B', 'Qwen 2.5 7B', 'GPT-4o', 'Gemma 2 9B', 'Gemini 2.5 Pro'],
    'x': [0.957, 0.871, 0.977, 0.947, 0.922, 0.906, 1.0],  # Abstract Audit Safety
    'y': [0.034, 0.030, 0.035, 0.088, 0.098, 0.409, 1.0],  # Real-World Refusal Rate
    'Type': ['Open Weights', 'Open Weights', 'Open Weights', 'Open Weights', 'API', 'Open Weights', 'API'],
    'Behavior': ['Contextual', 'Contextual', 'Contextual', 'Contextual', 'Contextual', 'Absolutist', 'Absolutist']
}
df = pd.DataFrame(data)

# 2. Setup the Plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.set_style("whitegrid")
sns.set_context("talk") # Increase general font size for better readability

# 3. Plot Points
# Define colors and markers specifically for clarity
palette = {'Contextual': '#34495e', 'Absolutist': '#e74c3c'}
markers = {'Open Weights': 'o', 'API': 'X'}

sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='Behavior',
    style='Type',
    s=400,               # Large, clear markers
    palette=palette,
    markers=markers,
    edgecolor='k',       # Thinner, cleaner edge
    linewidth=1.5,
    ax=ax
)

# 4. Labels and Title
ax.set_title("The Lexicographic Gap:\nContextual vs. Absolute Safety Models", fontsize=18, weight='bold', y=1.03)
ax.set_xlabel("Abstract Audit Safety (Lottery)", fontsize=14, labelpad=12)
ax.set_ylabel("Real-World Refusal Rate (OR-Bench)", fontsize=14, labelpad=12)

# Set axis limits with some padding
ax.set_xlim(0.85, 1.03)
ax.set_ylim(-0.05, 1.05)

# 5. Smart Labeling (Fixes Overlaps)
texts = []
for i in range(df.shape[0]):
    # Create the text annotation
    texts.append(ax.text(df.x[i], df.y[i], df.model[i], fontsize=12, weight='medium'))

# Use adjust_text to move labels away from markers and each other
adjust_text(texts,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.8),
            expand_points=(1.5, 1.5), # Push text further from points
            expand_text=(5, 5),   # Push text further from other text
            force_text=(0.1, 0.5),    # Force for text repulsion
            force_points=(0.2, 0.5)   # Force for point repulsion
           )

# 6. Clean up the Legend
# Get handles and labels, then re-create for full control
handles, labels = ax.get_legend_handles_labels()

# Create custom legend handles with cleaner styling (smaller size, no giant edges)
from matplotlib.lines import Line2D
custom_handles = [
    Line2D([0], [0], marker='o', color='w', label='Contextual', markerfacecolor=palette['Contextual'], markersize=12, markeredgecolor='k', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='w', label='Absolutist', markerfacecolor=palette['Absolutist'], markersize=12, markeredgecolor='k', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='k', label='Open Weights', markerfacecolor='w', markersize=12, markeredgewidth=1.5),
    Line2D([0], [0], marker='X', color='k', label='API', markerfacecolor='w', markersize=12, markeredgewidth=1.5),
]
custom_labels = ['Contextual', 'Absolutist', 'Open Weights', 'API']

# Add titles to the legend sections
legend_elements = [
    plt.Line2D([0], [0], color='none', label='Behavior'),
    custom_handles[0], custom_handles[1],
    plt.Line2D([0], [0], color='none', label=''), # Spacer
    plt.Line2D([0], [0], color='none', label='Type'),  
    custom_handles[2], custom_handles[3]
]

# Re-draw the legend with a cleaner look and better placement
ax.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.9, fancybox=True, fontsize=12, title_fontsize=13)

# Final layout adjustment
plt.tight_layout()

# Save the improved plot
plt.savefig("final_simplified_plot_improved.png", dpi=300, bbox_inches='tight')
plt.show()