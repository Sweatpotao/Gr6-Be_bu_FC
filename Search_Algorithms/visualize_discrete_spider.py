"""
Spider/Radar Chart Visualization for Discrete Algorithm Comparison
Based on algorithm_comparison_plan.md
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
with open('discrete_spider_data.json', 'r') as f:
    data = json.load(f)

# Categories for spider chart
categories = ['Solution Quality', 'Speed', 'Efficiency', 'Reliability', 'Convergence']
N = len(categories)

# Colors for algorithms - Blues for Classical algorithms
colors = {
    'BFS': '#1f77b4',      # Blue
    'DFS': '#2ca02c',       # Green  
    'UCS': '#ff7f0e',       # Orange
    'Greedy': '#9467bd',     # Purple
    'A*': '#d62728'          # Red
}

def create_spider_chart(problem_name, algorithms_data, save_path=None):
    """Create a spider chart for a specific problem."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Plot each algorithm
    for algo_name, values in algorithms_data.items():
        if algo_name in colors:
            values_closed = values + values[:1]  # Complete the circle
            ax.plot(angles, values_closed, 'o-', linewidth=2, 
                   label=algo_name, color=colors[algo_name], markersize=8)
            ax.fill(angles, values_closed, alpha=0.1, color=colors[algo_name])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=9)
    
    # Title and legend
    ax.set_title(f'{problem_name}\nAlgorithm Comparison', size=16, 
                fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), 
             fontsize=11, title='Algorithms', title_fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig

def create_combined_chart(all_data, save_path=None):
    """Create a combined spider chart with all problems."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), 
                            subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    problem_names = list(all_data.keys())
    
    for idx, (problem_name, algorithms_data) in enumerate(all_data.items()):
        ax = axes[idx]
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Plot each algorithm
        for algo_name, values in algorithms_data.items():
            if algo_name in colors:
                values_closed = values + values[:1]
                ax.plot(angles, values_closed, 'o-', linewidth=2, 
                       label=algo_name, color=colors[algo_name], markersize=6)
                ax.fill(angles, values_closed, alpha=0.1, color=colors[algo_name])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(problem_name, size=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend to the last subplot
    axes[-1].legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), 
                   fontsize=10, title='Algorithms', title_fontsize=11)
    
    plt.suptitle('Discrete Algorithm Comparison - Spider Charts', 
                size=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved combined chart: {save_path}")
    
    return fig

# Generate charts
print("Generating Spider Charts for Discrete Problems...")
print("=" * 60)

# Individual charts
for problem in data.keys():
    filename = problem.lower().replace('-', '_').replace(' ', '_') + '_spider.png'
    create_spider_chart(problem, data[problem], f'img/{filename}')

# Combined chart
create_combined_chart(data, 'img/discrete_spider_combined.png')

print("=" * 60)
print("All spider charts generated successfully!")
print("Charts saved to: Search_Algorithms/img/")
