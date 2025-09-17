import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_publication_ready_boxplots(results_dir, output_filename="publication_ready_boxplots.png"):
    """
    Create publication-ready box plots with professional styling.
    """
    # Load the detailed results
    detailed_results_path = os.path.join(results_dir, "detailed_resistance_results.csv")
    df_resistance = pd.read_csv(detailed_results_path)
    
    # Filter for 20 metrics
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        max_metrics = df_resistance["num_metrics"].max()
        df_20_metrics = df_resistance[df_resistance["num_metrics"] == max_metrics].dropna()
        metric_count = max_metrics
    else:
        metric_count = 20
    
    # Set publication-ready style
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    })
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Strategic Resistance Analysis ({metric_count} Metrics)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Define resistance measures
    measures_info = [
        ('bribery_resistance', 'Bribery Resistance', 'A'),
        ('manipulation_resistance', 'Manipulation Resistance', 'B'),
        ('deletion_resistance', 'Deletion Resistance', 'C'),
        ('cloning_resistance', 'Cloning Resistance', 'D')
    ]
    
    # Color scheme
    colors = ['#4472C4', '#E70011']  # Blue and red
    
    for idx, (measure, title, panel_label) in enumerate(measures_info):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create the boxplot
        bp = ax.boxplot([
            df_20_metrics[(df_20_metrics['elicitation'] == elicit) & 
                         (df_20_metrics['aggregation'] == 'arithmetic_mean')][measure].values
            for elicit in ['cumulative', 'fractional', 'approval', 'plurality']
        ] + [
            df_20_metrics[(df_20_metrics['elicitation'] == elicit) & 
                         (df_20_metrics['aggregation'] == 'median')][measure].values
            for elicit in ['cumulative', 'fractional', 'approval', 'plurality']
        ], 
        positions=[1, 2, 3, 4, 6, 7, 8, 9],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=2, color='black'),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.6)
        )
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            if i < 4:  # First 4 are arithmetic mean
                patch.set_facecolor(colors[0])
                patch.set_alpha(0.7)
            else:  # Last 4 are median
                patch.set_facecolor(colors[1])
                patch.set_alpha(0.7)
        
        # Customize axes
        ax.set_title(f'({panel_label}) {title}', fontweight='bold', pad=15)
        ax.set_ylabel('L1 Distance', fontweight='semibold')
        
        # Set x-axis labels
        ax.set_xticks([1, 2, 3, 4, 6, 7, 8, 9])
        ax.set_xticklabels(['Cum', 'Frac', 'App', 'Plur'] * 2, rotation=0)
        
        # Add group labels
        ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Arithmetic Mean', 
                ha='center', va='top', fontweight='bold', color=colors[0])
        ax.text(7.5, ax.get_ylim()[1] * 0.95, 'Median', 
                ha='center', va='top', fontweight='bold', color=colors[1])
        
        # Add vertical separator
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set x-axis limits
        ax.set_xlim(0, 10)
    
    # Add overall legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.7, label='Arithmetic Mean'),
        Patch(facecolor=colors[1], alpha=0.7, label='Median')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, frameon=False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    # Save the plot
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Publication-ready box plots saved to: {output_path}")
    
    plt.show()
    return fig

def create_horizontal_boxplots(results_dir, output_filename="horizontal_resistance_boxplots.png"):
    """
    Create horizontal box plots for better label readability.
    """
    # Load the detailed results
    detailed_results_path = os.path.join(results_dir, "detailed_resistance_results.csv")
    df_resistance = pd.read_csv(detailed_results_path)
    
    # Filter for 20 metrics
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        max_metrics = df_resistance["num_metrics"].max()
        df_20_metrics = df_resistance[df_resistance["num_metrics"] == max_metrics].dropna()
        metric_count = max_metrics
    else:
        metric_count = 20
    
    # Create combined labels for better organization
    df_20_metrics['method_combo'] = (df_20_metrics['elicitation'].str.capitalize() + 
                                   ' + ' + df_20_metrics['aggregation'].str.replace('arithmetic_mean', 'Mean'))
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Strategic Resistance Analysis ({metric_count} Metrics)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    measures_info = [
        ('bribery_resistance', 'Bribery Resistance'),
        ('manipulation_resistance', 'Manipulation Resistance'),
        ('deletion_resistance', 'Deletion Resistance'),
        ('cloning_resistance', 'Cloning Resistance')
    ]
    
    for idx, (measure, title) in enumerate(measures_info):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create horizontal boxplot
        sns.boxplot(
            data=df_20_metrics,
            y='method_combo',
            x=measure,
            ax=ax,
            orient='h',
            palette='Set2',
            linewidth=1.2,
            fliersize=3
        )
        
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('L1 Distance', fontweight='semibold')
        ax.set_ylabel('')
        
        # Grid
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # Adjust tick labels
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # Save the plot
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Horizontal box plots saved to: {output_path}")
    
    plt.show()
    return fig

def main():
    """
    Create multiple clean visualization options.
    """
    results_dir = "/Users/idrees/Code/feature-based-voting/results/metric_variety_social_welfare/run_20250913_144510"
    
    print("Creating publication-ready box plots...")
    fig1 = create_publication_ready_boxplots(results_dir)
    
    print("\nCreating horizontal box plots...")
    fig2 = create_horizontal_boxplots(results_dir)
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()
