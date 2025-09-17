import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.voting_model import VotingSimulator, ElicitationMethod
from models.optimizers import bribery_optimization, manipulation, control_by_cloning, control_by_deletion
from scripts import config
from scripts.util import save_simulation_results

def calculate_social_welfare(votes, value_matrix, aggregation_method, elicitation_method):
    """
    Calculate social welfare for each agent.
    
    Social welfare = L1 distance between:
    - Rule output × project values 
    - Agent's importance × project values
    
    Args:
        votes: Array of shape (num_voters, num_metrics) - voting matrix
        value_matrix: Array of shape (num_projects, num_metrics) - project values
        aggregation_method: "arithmetic_mean" or "median"
        elicitation_method: ElicitationMethod enum
        
    Returns:
        Array of social welfare values for each agent
    """
    num_voters, num_metrics = votes.shape
    num_projects = value_matrix.shape[0]
    
    # Compute aggregated importance weights
    if aggregation_method == "arithmetic_mean":
        aggregated_weights = np.mean(votes, axis=0)
    elif aggregation_method == "median":
        aggregated_weights = np.median(votes, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Compute project scores using aggregated weights
    aggregated_scores = value_matrix @ aggregated_weights
    
    # Compute social welfare for each agent
    social_welfare_per_agent = []
    
    for agent_idx in range(num_voters):
        # Agent's individual importance weights
        agent_weights = votes[agent_idx, :]
        
        # Project scores using agent's individual weights
        agent_scores = value_matrix @ agent_weights
        
        # L1 distance between aggregated scores and agent's preferred scores
        l1_distance = np.sum(np.abs(aggregated_scores - agent_scores))
        social_welfare_per_agent.append(l1_distance)
    
    return np.array(social_welfare_per_agent)

def run_metric_variety_experiment():
    """
    Run experiment with varying number of metrics and social welfare analysis.
    """
    # Configuration
    num_instances = 50
    num_voters = 100
    num_projects = 200
    metric_counts = [5, 10, 15, 20, 25, 30]  # Different numbers of metrics
    elicitation_methods = [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, 
                          ElicitationMethod.APPROVAL, ElicitationMethod.PLURALITY]
    aggregation_methods = ["arithmetic_mean", "median"]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "metric_variety_social_welfare", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Storage for results
    all_results = []
    social_welfare_results = []
    
    print("Running Metric Variety and Social Welfare Experiment...")
    print(f"Testing {len(metric_counts)} different metric counts: {metric_counts}")
    print(f"Testing {len(elicitation_methods)} elicitation methods")
    print(f"Running {num_instances} instances per configuration")
    
    for metric_count in metric_counts:
        print(f"\n--- Testing with {metric_count} metrics ---")
        
        # Generate metrics names
        metrics = [f"metric_{i}" for i in range(metric_count)]
        
        for elicitation in elicitation_methods:
            print(f"  Testing {elicitation.value} elicitation...")
            
            for instance in range(num_instances):
                if instance % 10 == 0:
                    print(f"    Instance {instance + 1}/{num_instances}")
                
                # Create simulator
                simulator = VotingSimulator(
                    num_voters=num_voters,
                    num_projects=num_projects,
                    metrics=metrics,
                    elicitation_method=elicitation,
                    alpha=config.alpha
                )
                
                # Generate data
                votes = simulator.generate_votes()
                value_matrix = simulator.generate_value_matrix()
                ideal_scores = simulator.generate_ideal_scores()
                
                # Normalize value matrix
                value_matrix = simulator.normalize_value_matrix(value_matrix)
                
                # Test each aggregation method
                for aggregation in aggregation_methods:
                    try:
                        # Calculate social welfare
                        social_welfare = calculate_social_welfare(
                            votes, value_matrix, aggregation, elicitation
                        )
                        mean_social_welfare = np.mean(social_welfare)
                        
                        # Run resistance tests
                        l1_bribery = bribery_optimization(
                            votes, value_matrix, ideal_scores, 
                            config.bribery_budget, elicitation, aggregation
                        )
                        
                        l1_manipulation = manipulation(
                            votes, value_matrix, elicitation, aggregation
                        )
                        
                        l1_deletion = control_by_deletion(
                            votes, value_matrix, ideal_scores, 
                            config.deletion_budget, elicitation, aggregation
                        )
                        
                        l1_cloning = control_by_cloning(
                            votes, value_matrix, ideal_scores, 
                            config.cloning_budget, elicitation, aggregation
                        )
                        
                        # Store resistance results
                        all_results.append({
                            "num_metrics": metric_count,
                            "elicitation": elicitation.value,
                            "aggregation": aggregation,
                            "instance": instance,
                            "bribery_resistance": l1_bribery,
                            "manipulation_resistance": l1_manipulation,
                            "deletion_resistance": l1_deletion,
                            "cloning_resistance": l1_cloning
                        })
                        
                        # Store social welfare results
                        social_welfare_results.append({
                            "num_metrics": metric_count,
                            "elicitation": elicitation.value,
                            "aggregation": aggregation,
                            "instance": instance,
                            "mean_social_welfare": mean_social_welfare,
                            "social_welfare_std": np.std(social_welfare),
                            "social_welfare_min": np.min(social_welfare),
                            "social_welfare_max": np.max(social_welfare)
                        })
                        
                    except Exception as e:
                        print(f"    Error in {elicitation.value} + {aggregation}: {e}")
                        # Store failed results with NaN values
                        all_results.append({
                            "num_metrics": metric_count,
                            "elicitation": elicitation.value,
                            "aggregation": aggregation,
                            "instance": instance,
                            "bribery_resistance": np.nan,
                            "manipulation_resistance": np.nan,
                            "deletion_resistance": np.nan,
                            "cloning_resistance": np.nan
                        })
                        
                        social_welfare_results.append({
                            "num_metrics": metric_count,
                            "elicitation": elicitation.value,
                            "aggregation": aggregation,
                            "instance": instance,
                            "mean_social_welfare": np.nan,
                            "social_welfare_std": np.nan,
                            "social_welfare_min": np.nan,
                            "social_welfare_max": np.nan
                        })
    
    # Convert to DataFrames
    df_resistance = pd.DataFrame(all_results)
    df_social_welfare = pd.DataFrame(social_welfare_results)
    
    # Save detailed results
    df_resistance.to_csv(os.path.join(results_dir, "detailed_resistance_results.csv"), index=False)
    df_social_welfare.to_csv(os.path.join(results_dir, "detailed_social_welfare_results.csv"), index=False)
    
    # Create visualizations
    create_social_welfare_histogram(df_social_welfare, results_dir)
    create_resistance_boxplots_20_metrics(df_resistance, results_dir)
    
    # Save summary statistics
    summary_resistance = df_resistance.groupby(["num_metrics", "elicitation", "aggregation"]).mean()
    summary_social_welfare = df_social_welfare.groupby(["num_metrics", "elicitation", "aggregation"]).mean()
    
    summary_resistance.to_csv(os.path.join(results_dir, "summary_resistance_results.csv"))
    summary_social_welfare.to_csv(os.path.join(results_dir, "summary_social_welfare_results.csv"))
    
    print(f"\nExperiment completed! Results saved to: {results_dir}")
    return df_resistance, df_social_welfare

def create_social_welfare_histogram(df_social_welfare, results_dir):
    """
    Create histogram showing social welfare vs number of metrics.
    Two bars per metric count: one for median, one for mean aggregation.
    """
    # Filter out NaN values
    df_clean = df_social_welfare.dropna()
    
    # Calculate mean social welfare for each metric count and aggregation method
    summary = df_clean.groupby(["num_metrics", "aggregation"])["mean_social_welfare"].mean().reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique metric counts and sort them
    metric_counts = sorted(df_clean["num_metrics"].unique())
    x_pos = np.arange(len(metric_counts))
    width = 0.35
    
    # Separate data for mean and median
    mean_data = summary[summary["aggregation"] == "arithmetic_mean"]["mean_social_welfare"].values
    median_data = summary[summary["aggregation"] == "median"]["mean_social_welfare"].values
    
    # Create bars
    plt.bar(x_pos - width/2, mean_data, width, label='Arithmetic Mean', alpha=0.8, color='skyblue')
    plt.bar(x_pos + width/2, median_data, width, label='Median', alpha=0.8, color='lightcoral')
    
    # Customize plot
    plt.xlabel('Number of Metrics', fontsize=12)
    plt.ylabel('Mean Social Welfare (L1 Distance)', fontsize=12)
    plt.title('Social Welfare vs Number of Metrics\n(Mean over all instances and elicitation methods)', fontsize=14)
    plt.xticks(x_pos, metric_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean_val, median_val) in enumerate(zip(mean_data, median_data)):
        plt.text(i - width/2, mean_val + 0.01, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, median_val + 0.01, f'{median_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "social_welfare_histogram.png"), dpi=300, bbox_inches='tight')
    plt.show()

def create_resistance_boxplots_20_metrics(df_resistance, results_dir):
    """
    Create box plots for resistance measures with 20 metrics.
    """
    # Filter for 20 metrics only
    df_20_metrics = df_resistance[df_resistance["num_metrics"] == 20].dropna()
    
    if df_20_metrics.empty:
        print("No data available for 20 metrics. Creating plots with available data...")
        # Use the highest available metric count
        max_metrics = df_resistance["num_metrics"].max()
        df_20_metrics = df_resistance[df_resistance["num_metrics"] == max_metrics].dropna()
        print(f"Using {max_metrics} metrics instead of 20.")
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Resistance Measures with 20 Metrics\n(Box plots across all instances and elicitation methods)', fontsize=16)
    
    # Define resistance measures and their positions
    resistance_measures = [
        ('bribery_resistance', 'Bribery Resistance'),
        ('manipulation_resistance', 'Manipulation Resistance'),
        ('deletion_resistance', 'Deletion Resistance'),
        ('cloning_resistance', 'Cloning Resistance')
    ]
    
    for idx, (measure, title) in enumerate(resistance_measures):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create box plot
        sns.boxplot(data=df_20_metrics, x='elicitation', y=measure, hue='aggregation', ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Elicitation Method', fontsize=10)
        ax.set_ylabel('L1 Distance', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "resistance_boxplots_20_metrics.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df_resistance, df_social_welfare = run_metric_variety_experiment()
    
    print("\nExperiment Summary:")
    print(f"Total resistance test results: {len(df_resistance)}")
    print(f"Total social welfare results: {len(df_social_welfare)}")
    print(f"Metric counts tested: {sorted(df_resistance['num_metrics'].unique())}")
    print(f"Elicitation methods tested: {df_resistance['elicitation'].unique()}")
    print(f"Aggregation methods tested: {df_resistance['aggregation'].unique()}")
