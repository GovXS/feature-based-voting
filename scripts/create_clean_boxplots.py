import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_clean_resistance_boxplots(results_dir, output_filename="clean_resistance_boxplots_20_metrics.png"):
    """
    Create clean box plots for resistance measures with 20 metrics.
    Addresses overlapping text issues by:
    1. Using larger figure size
    2. Rotating x-axis labels
    3. Adjusting spacing and margins
    4. Using abbreviated labels
    5. Improving color scheme
    """
    # Load the detailed results
    detailed_results_path = os.path.join(results_dir, "detailed_resistance_results.csv")
    df_resistance = pd.read_csv(detailed_results_path)
    
    # Filter for 20 metrics only
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        print("No data available for 20 metrics. Using the highest available metric count...")
        max_metrics = df_resistance["num_metrics"].max()
        df_20_metrics = df_resistance[df_resistance["num_metrics"] == max_metrics].dropna()
        print(f"Using {max_metrics} metrics instead of 20.")
        metric_count = max_metrics
    else:
        metric_count = 20
    
    # Create abbreviated labels for cleaner display
    elicitation_labels = {
        'cumulative': 'Cumul.',
        'fractional': 'Fract.',
        'approval': 'Approv.',
        'plurality': 'Plural.'
    }
    
    aggregation_labels = {
        'arithmetic_mean': 'Mean',
        'median': 'Median'
    }
    
    # Apply label mapping
    df_20_metrics = df_20_metrics.copy()
    df_20_metrics['elicitation_short'] = df_20_metrics['elicitation'].map(elicitation_labels)
    df_20_metrics['aggregation_short'] = df_20_metrics['aggregation'].map(aggregation_labels)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the plot with larger size and better spacing
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Resistance Measures with {metric_count} Metrics\n(Distribution across all instances)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define resistance measures and their positions
    resistance_measures = [
        ('bribery_resistance', 'Bribery Resistance'),
        ('manipulation_resistance', 'Manipulation Resistance'),
        ('deletion_resistance', 'Deletion Resistance'),
        ('cloning_resistance', 'Cloning Resistance')
    ]
    
    # Color palette for better distinction
    colors = ['lightblue', 'lightcoral']
    
    for idx, (measure, title) in enumerate(resistance_measures):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create box plot with improved styling
        box_plot = sns.boxplot(
            data=df_20_metrics, 
            x='elicitation_short', 
            y=measure, 
            hue='aggregation_short', 
            ax=ax,
            palette=colors,
            linewidth=1.2,
            fliersize=3
        )
        
        # Customize the subplot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Elicitation Method', fontsize=12, fontweight='semibold')
        ax.set_ylabel('L1 Distance', fontsize=12, fontweight='semibold')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=0, labelsize=11)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Improve legend
        if ax.get_legend():
            ax.legend(title='Aggregation', fontsize=10, title_fontsize=11, 
                     loc='upper right', framealpha=0.9)
        
        # Add some statistical information as text
        if measure in ['bribery_resistance', 'manipulation_resistance']:
            # Add median values as text annotations
            medians = df_20_metrics.groupby(['elicitation_short', 'aggregation_short'])[measure].median()
            y_max = df_20_metrics[measure].max()
            y_pos = y_max * 0.9
            
            for i, elicitation in enumerate(df_20_metrics['elicitation_short'].unique()):
                for j, (agg, color) in enumerate(zip(['Mean', 'Median'], colors)):
                    if (elicitation, agg) in medians.index:
                        median_val = medians[(elicitation, agg)]
                        ax.text(i + (j-0.5)*0.2, y_pos, f'{median_val:.1f}', 
                               ha='center', va='bottom', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save the plot
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Clean box plots saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def create_summary_statistics_table(results_dir):
    """
    Create a summary statistics table for the resistance measures.
    """
    # Load the detailed results
    detailed_results_path = os.path.join(results_dir, "detailed_resistance_results.csv")
    df_resistance = pd.read_csv(detailed_results_path)
    
    # Filter for 20 metrics
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        max_metrics = df_resistance["num_metrics"].max()
        df_20_metrics = df_resistance[df_resistance["num_metrics"] == max_metrics].dropna()
        print(f"Using {max_metrics} metrics for summary statistics.")
    
    # Calculate summary statistics
    summary_stats = df_20_metrics.groupby(['elicitation', 'aggregation']).agg({
        'bribery_resistance': ['mean', 'median', 'std'],
        'manipulation_resistance': ['mean', 'median', 'std'],
        'deletion_resistance': ['mean', 'median', 'std'],
        'cloning_resistance': ['mean', 'median', 'std']
    }).round(2)
    
    # Save to CSV
    summary_path = os.path.join(results_dir, "resistance_summary_statistics_20_metrics.csv")
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    return summary_stats

def main():
    """
    Main function to create clean visualizations from existing results.
    """
    # Path to the results directory
    results_dir = "/Users/idrees/Code/feature-based-voting/results/metric_variety_social_welfare/run_20250913_144510"
    
    print("Creating clean box plots...")
    fig = create_clean_resistance_boxplots(results_dir)
    
    print("\nCreating summary statistics table...")
    summary_stats = create_summary_statistics_table(results_dir)
    
    print("\nSummary Statistics Preview:")
    print(summary_stats.head())
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
