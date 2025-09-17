import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from models.voting_model import VotingSimulator, ElicitationMethod
from models.optimizers import bribery_optimization, manipulation, control_by_cloning, control_by_deletion
from scripts import config
from scripts.util import save_simulation_results, visualize_combinations_experiment_results
from scripts.data_loader import load_mechanical_turk_data, validate_and_normalize_votes, get_available_data_files

def transform_cumulative_to_approval(votes: np.ndarray) -> np.ndarray:
    """
    Transform cumulative voting data to approval voting.
    Any vote > 0 becomes 1, votes = 0 stay 0.
    
    Args:
        votes: Array of cumulative votes (each row sums to 1)
    
    Returns:
        Array of approval votes (binary: 0 or 1)
    """
    approval_votes = np.where(votes > 0, 1, 0).astype(float)
    return approval_votes

# Configuration for real data experiments
aggregation_methods = ["arithmetic_mean", "median"]

# Load real voting data
print("Available data files:")
data_files = get_available_data_files()
for i, file in enumerate(data_files):
    print(f"{i}: {file}")

# Use the first available file (you can modify this to select a specific file)
if not data_files:
    raise FileNotFoundError("No mechanical turk data files found in data directory")

data_file = data_files[0]  # Using first file
print(f"\nLoading data from: {data_file}")

# Load the real voting data
real_votes_raw, voter_ids, metric_ids = load_mechanical_turk_data(data_file)

print(f"\nLoaded real voting data:")
print(f"- Number of voters: {len(voter_ids)}")
print(f"- Number of metrics: {len(metric_ids)}")
print(f"- Voting matrix shape: {real_votes_raw.shape}")

# Validate that the data is already in cumulative format
real_votes_cumulative = validate_and_normalize_votes(real_votes_raw, "cumulative")

# Transform to approval voting
real_votes_approval = transform_cumulative_to_approval(real_votes_cumulative)

print(f"\nData transformation:")
print(f"- Cumulative votes (first voter): {real_votes_cumulative[0]}")
print(f"- Approval votes (first voter): {real_votes_approval[0]}")
print(f"- Cumulative row sum: {np.sum(real_votes_cumulative[0])}")
print(f"- Approval row sum: {np.sum(real_votes_approval[0])}")

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "cumulative_vs_approval_real_data", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Storage for final results
resistance_results = []

# Test both elicitation methods
elicitation_methods = [
    (ElicitationMethod.CUMULATIVE, real_votes_cumulative, "cumulative"),
    (ElicitationMethod.APPROVAL, real_votes_approval, "approval")
]

for elicitation_method, votes_data, method_name in elicitation_methods:
    print(f"\nRunning simulations for {method_name.upper()} elicitation with real data...\n")

    for instance in range(min(config.num_instances, 50)):  # 50 instances for each method
        print(f"Running {method_name} instance {instance + 1}/50...")

        # Create simulator with real voting data
        simulator = VotingSimulator(
            num_voters=votes_data.shape[0],  # Use actual number of voters from real data
            num_projects=config.num_projects,  # Keep synthetic projects
            metrics=metric_ids,  # Use actual metric IDs from real data
            elicitation_method=elicitation_method,
            alpha=config.alpha
        )
        
        # Set the real voting data
        simulator.set_real_votes(votes_data)
        
        # Generate votes (will return real data)
        votes = simulator.generate_votes()
        
        # Generate synthetic value matrix and ideal scores (since we don't have real project data)
        value_matrix = simulator.generate_value_matrix()
        ideal_scores = simulator.generate_ideal_scores()

        # Evaluate all problems for each aggregation method
        for aggregation in aggregation_methods:
            print(f"  Testing {aggregation} aggregation...")

            try:
                # Bribery Optimization
                l1_bribery = bribery_optimization(votes, value_matrix, ideal_scores, config.bribery_budget, elicitation_method, aggregation)

                # Manipulation
                l1_manipulation = manipulation(votes, value_matrix, elicitation_method, aggregation)

                # Feature Deletion Control
                l1_deletion = control_by_deletion(votes, value_matrix, ideal_scores, config.deletion_budget, elicitation_method, aggregation)

                # Feature Cloning Control
                l1_cloning = control_by_cloning(votes, value_matrix, ideal_scores, config.cloning_budget, elicitation_method, aggregation)

                # Store results
                resistance_results.append({
                    "elicitation": method_name,
                    "aggregation": aggregation,
                    "instance": instance,
                    "num_real_voters": votes_data.shape[0],
                    "num_metrics": len(metric_ids),
                    "bribery_resistance": l1_bribery,
                    "manipulation_resistance": l1_manipulation,
                    "deletion_resistance": l1_deletion,
                    "cloning_resistance": l1_cloning
                })
                
            except Exception as e:
                print(f"    Error in {aggregation} aggregation: {e}")
                # Store failed results with NaN values
                resistance_results.append({
                    "elicitation": method_name,
                    "aggregation": aggregation,
                    "instance": instance,
                    "num_real_voters": votes_data.shape[0],
                    "num_metrics": len(metric_ids),
                    "bribery_resistance": np.nan,
                    "manipulation_resistance": np.nan,
                    "deletion_resistance": np.nan,
                    "cloning_resistance": np.nan
                })

# Convert results to DataFrame and compute average L1 distances
df_results = pd.DataFrame(resistance_results)

# Remove NaN results before computing summary
df_results_clean = df_results.dropna()
df_summary = df_results_clean.groupby(["elicitation", "aggregation"]).mean()

# Save detailed and summary results
df_results.to_csv(os.path.join(results_dir, "detailed_results_cumulative_vs_approval.csv"), index=False)
df_summary.to_csv(os.path.join(results_dir, "summary_results_cumulative_vs_approval.csv"))

# Save both voting matrices for analysis
np.savetxt(os.path.join(results_dir, "votes_cumulative.csv"), real_votes_cumulative, delimiter=",")
np.savetxt(os.path.join(results_dir, "votes_approval.csv"), real_votes_approval, delimiter=",")

# Save transformation comparison
comparison_data = {
    "data_file": data_file,
    "num_real_voters": real_votes_cumulative.shape[0],
    "num_metrics": len(metric_ids),
    "metric_ids": metric_ids,
    "transformation_stats": {
        "cumulative_votes_mean": np.mean(real_votes_cumulative),
        "cumulative_votes_std": np.std(real_votes_cumulative),
        "approval_votes_mean": np.mean(real_votes_approval),
        "approval_votes_std": np.std(real_votes_approval),
        "percentage_nonzero_cumulative": np.mean(real_votes_cumulative > 0) * 100,
        "percentage_nonzero_approval": np.mean(real_votes_approval > 0) * 100,
        "avg_votes_per_voter_cumulative": np.mean(np.sum(real_votes_cumulative > 0, axis=1)),
        "avg_votes_per_voter_approval": np.mean(np.sum(real_votes_approval > 0, axis=1))
    }
}

# Save metadata about the experiment
sim_params = {
    "metrics": metric_ids,
    "num_voters": real_votes_cumulative.shape[0],
    "num_projects": config.num_projects,
    "budget": config.bribery_budget,
    "cloning_budget": config.cloning_budget,
    "deletion_budget": config.deletion_budget,
    "using_real_data": True,
    "elicitation_methods": ["cumulative", "approval"],
    "comparison_data": comparison_data
}

save_simulation_results(results_dir, sim_params, real_votes_cumulative, value_matrix, ideal_scores, df_results)

# Print summary results
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Results saved to: {results_dir}")
print(f"Total experiments run: {len(df_results_clean)}")
print(f"Failed experiments: {len(df_results) - len(df_results_clean)}")

print("\nSummary Results (Mean L1 Distances):")
print(df_summary.round(4))

print("\nData Transformation Summary:")
for key, value in comparison_data["transformation_stats"].items():
    print(f"- {key}: {value:.4f}")

# Create visualization (if the function exists)
try:
    visualize_combinations_experiment_results(results_dir, df_summary.reset_index(), df_results_clean)
    print(f"\nVisualization saved to {results_dir}")
except Exception as e:
    print(f"\nVisualization failed: {e}")

print("\nExperiment completed successfully!") 