import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_results(results_dir):
    """Load the detailed results from CSV files."""
    social_welfare_path = os.path.join(results_dir, "detailed_social_welfare_results.csv")
    resistance_path = os.path.join(results_dir, "detailed_resistance_results.csv")
    
    df_social_welfare = pd.read_csv(social_welfare_path)
    df_resistance = pd.read_csv(resistance_path)
    
    return df_social_welfare, df_resistance

def create_enhanced_social_welfare_histogram(df_social_welfare, save_dir):
    """
    Create enhanced social welfare histogram with better visualization.
    """
    # Filter out NaN values
    df_clean = df_social_welfare.dropna()
    
    # Calculate mean social welfare for each metric count and aggregation method
    summary = df_clean.groupby(["num_metrics", "aggregation"]).agg({
        "mean_social_welfare": ["mean", "std", "count"]
    }).reset_index()
    
    # Flatten column names
    summary.columns = ["num_metrics", "aggregation", "mean_sw", "std_sw", "count"]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Get unique metric counts and sort them
    metric_counts = sorted(df_clean["num_metrics"].unique())
    x_pos = np.arange(len(metric_counts))
    width = 0.35
    
    # Separate data for mean and median
    mean_data = summary[summary["aggregation"] == "arithmetic_mean"]
    median_data = summary[summary["aggregation"] == "median"]
    
    # Get values and standard errors
    mean_values = mean_data["mean_sw"].values
    mean_errors = mean_data["std_sw"].values / np.sqrt(mean_data["count"].values)
    
    median_values = median_data["mean_sw"].values
    median_errors = median_data["std_sw"].values / np.sqrt(median_data["count"].values)
    
    # Create bars with error bars
    bars1 = plt.bar(x_pos - width/2, mean_values, width, 
                   yerr=mean_errors, capsize=5,
                   label='Arithmetic Mean', alpha=0.8, color='skyblue', 
                   edgecolor='navy', linewidth=1.2)
    bars2 = plt.bar(x_pos + width/2, median_values, width,
                   yerr=median_errors, capsize=5,
                   label='Median', alpha=0.8, color='lightcoral',
                   edgecolor='darkred', linewidth=1.2)
    
    # Customize plot
    plt.xlabel('Number of Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Social Welfare (L1 Distance)', fontsize=14, fontweight='bold')
    plt.title('Social Welfare vs Number of Metrics\n(Mean ± SEM over all instances and elicitation methods)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x_pos, metric_counts, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (mean_val, median_val) in enumerate(zip(mean_values, median_values)):
        plt.text(i - width/2, mean_val + mean_errors[i] + 0.5, f'{mean_val:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(i + width/2, median_val + median_errors[i] + 0.5, f'{median_val:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add trend lines
    z_mean = np.polyfit(metric_counts, mean_values, 1)
    p_mean = np.poly1d(z_mean)
    plt.plot(x_pos, p_mean(metric_counts), "--", alpha=0.6, color='navy', linewidth=2)
    
    z_median = np.polyfit(metric_counts, median_values, 1)
    p_median = np.poly1d(z_median)
    plt.plot(x_pos, p_median(metric_counts), "--", alpha=0.6, color='darkred', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "enhanced_social_welfare_histogram.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary

def create_resistance_boxplots_20_metrics(df_resistance, save_dir):
    """
    Create box plots for resistance measures with 20 metrics.
    """
    # Filter for 20 metrics only
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        print("No data available for 20 metrics.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Resistance Measures with 20 Metrics', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define resistance measures and their positions
    resistance_measures = [
        ('bribery_resistance', 'Bribery Resistance'),
        ('manipulation_resistance', 'Manipulation Resistance'),
        ('deletion_resistance', 'Deletion Resistance'),
        ('cloning_resistance', 'Cloning Resistance')
    ]
    
    # Color palette
    colors = ['lightblue', 'lightcoral']
    
    for idx, (measure, title) in enumerate(resistance_measures):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create box plot with custom styling
        box_plot = sns.boxplot(data=df_20_metrics, x='elicitation', y=measure, 
                              hue='aggregation', ax=ax, palette=colors)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Elicitation Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('L1 Distance', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Style the legend
        if ax.get_legend():
            ax.get_legend().set_title('Aggregation Method', prop={'weight': 'bold'})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, "enhanced_resistance_boxplots_20_metrics.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_social_welfare_by_elicitation_heatmap(df_social_welfare, save_dir):
    """
    Create a heatmap showing social welfare across metrics and elicitation methods.
    """
    # Calculate mean social welfare by metric count and elicitation method
    pivot_data = df_social_welfare.groupby(['num_metrics', 'elicitation', 'aggregation'])['mean_social_welfare'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, agg_method in enumerate(['arithmetic_mean', 'median']):
        data_subset = pivot_data[pivot_data['aggregation'] == agg_method]
        heatmap_data = data_subset.pivot(index='elicitation', columns='num_metrics', values='mean_social_welfare')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[idx], cbar_kws={'label': 'Mean Social Welfare'})
        axes[idx].set_title(f'Social Welfare - {agg_method.replace("_", " ").title()}', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Elicitation Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "social_welfare_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()

def create_resistance_trends_plot(df_resistance, save_dir):
    """
    Create line plots showing resistance trends across different metric counts.
    """
    # Calculate mean resistance by metric count
    resistance_summary = df_resistance.groupby(['num_metrics', 'aggregation']).agg({
        'bribery_resistance': 'mean',
        'manipulation_resistance': 'mean',
        'deletion_resistance': 'mean',
        'cloning_resistance': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Resistance Trends Across Different Numbers of Metrics', 
                 fontsize=16, fontweight='bold')
    
    resistance_measures = ['bribery_resistance', 'manipulation_resistance', 
                          'deletion_resistance', 'cloning_resistance']
    titles = ['Bribery Resistance', 'Manipulation Resistance', 
              'Deletion Resistance', 'Cloning Resistance']
    
    for idx, (measure, title) in enumerate(zip(resistance_measures, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        for agg_method in ['arithmetic_mean', 'median']:
            data = resistance_summary[resistance_summary['aggregation'] == agg_method]
            ax.plot(data['num_metrics'], data[measure], 'o-', 
                   label=agg_method.replace('_', ' ').title(), linewidth=2, markersize=6)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Metrics', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Resistance (L1 Distance)', fontsize=10, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "resistance_trends.png"), dpi=300, bbox_inches='tight')
    plt.show()

def create_social_welfare_distribution_analysis(df_social_welfare, save_dir):
    """
    Create violin plots showing the distribution of social welfare across different metrics.
    """
    plt.figure(figsize=(16, 8))
    
    # Create subplot for each aggregation method
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for idx, agg_method in enumerate(['arithmetic_mean', 'median']):
        data_subset = df_social_welfare[df_social_welfare['aggregation'] == agg_method]
        
        sns.violinplot(data=data_subset, x='num_metrics', y='mean_social_welfare', 
                      hue='elicitation', ax=axes[idx], inner='box')
        
        axes[idx].set_title(f'Social Welfare Distribution - {agg_method.replace("_", " ").title()}', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Social Welfare (L1 Distance)', fontsize=12, fontweight='bold')
        axes[idx].legend(title='Elicitation Method', title_fontsize=10, fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "social_welfare_distributions.png"), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(df_social_welfare, df_resistance, save_dir):
    """
    Generate and save summary statistics tables.
    """
    # Social welfare summary
    sw_summary = df_social_welfare.groupby(['num_metrics', 'aggregation']).agg({
        'mean_social_welfare': ['mean', 'std', 'min', 'max'],
        'social_welfare_std': 'mean'
    }).round(3)
    
    # Resistance summary
    resistance_summary = df_resistance.groupby(['num_metrics', 'aggregation']).agg({
        'bribery_resistance': ['mean', 'std'],
        'manipulation_resistance': ['mean', 'std'],
        'deletion_resistance': ['mean', 'std'],
        'cloning_resistance': ['mean', 'std']
    }).round(3)
    
    # Save summaries
    sw_summary.to_csv(os.path.join(save_dir, "social_welfare_summary_stats.csv"))
    resistance_summary.to_csv(os.path.join(save_dir, "resistance_summary_stats.csv"))
    
    print("Summary statistics saved!")
    print("\nSocial Welfare Summary (first few rows):")
    print(sw_summary.head())
    print("\nResistance Summary (first few rows):")
    print(resistance_summary.head())

def main():
    """Main analysis function."""
    results_dir = "/Users/idrees/Code/feature-based-voting/results/metric_variety_social_welfare/run_20250913_144510"
    save_dir = results_dir  # Save plots in the same directory
    
    print("Loading results...")
    df_social_welfare, df_resistance = load_results(results_dir)
    
    print(f"Loaded {len(df_social_welfare)} social welfare records")
    print(f"Loaded {len(df_resistance)} resistance records")
    print(f"Metric counts: {sorted(df_social_welfare['num_metrics'].unique())}")
    print(f"Elicitation methods: {df_social_welfare['elicitation'].unique()}")
    print(f"Aggregation methods: {df_social_welfare['aggregation'].unique()}")
    
    print("\n1. Creating enhanced social welfare histogram...")
    #summary = create_enhanced_social_welfare_histogram(df_social_welfare, save_dir)
    
    print("\n2. Creating resistance boxplots for 20 metrics...")
    create_resistance_boxplots_20_metrics(df_resistance, save_dir)
    
    print("\n3. Creating social welfare heatmap...")
    #create_social_welfare_by_elicitation_heatmap(df_social_welfare, save_dir)
    
    print("\n4. Creating resistance trends plot...")
    #create_resistance_trends_plot(df_resistance, save_dir)
    
    print("\n5. Creating social welfare distribution analysis...")
    create_social_welfare_distribution_analysis(df_social_welfare, save_dir)
    
    print("\n6. Generating summary statistics...")
    generate_summary_statistics(df_social_welfare, df_resistance, save_dir)
    
    print(f"\nAll analysis complete! Plots saved to: {save_dir}")

if __name__ == "__main__":
    main()
