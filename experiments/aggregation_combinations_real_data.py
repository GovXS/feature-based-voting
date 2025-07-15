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

# Configuration for real data experiments
# The real data is already in cumulative format (each row sums to 1)
elicitation_method = ElicitationMethod.CUMULATIVE
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
print(f"- Elicitation method: CUMULATIVE (real data format)")

# Validate that the data is already in cumulative format
real_votes = validate_and_normalize_votes(real_votes_raw, "cumulative")

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "aggregation_combinations_real_data", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Storage for final results
resistance_results = []

print(f"\nRunning simulations for CUMULATIVE elicitation with real data...\n")

for instance in range(min(config.num_instances, 50)):  # Increase instances since we only test one elicitation
    print(f"Running instance {instance + 1}/50...")

    # Create simulator with real voting data
    simulator = VotingSimulator(
        num_voters=real_votes.shape[0],  # Use actual number of voters from real data
        num_projects=config.num_projects,  # Keep synthetic projects
        metrics=metric_ids,  # Use actual metric IDs from real data
        elicitation_method=elicitation_method,
        alpha=config.alpha
    )
    
    # Set the real voting data
    simulator.set_real_votes(real_votes)
    
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
                "elicitation": elicitation_method.value,
                "aggregation": aggregation,
                "instance": instance,
                "num_real_voters": real_votes.shape[0],
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
                "elicitation": elicitation_method.value,
                "aggregation": aggregation,
                "instance": instance,
                "num_real_voters": real_votes.shape[0],
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
df_results.to_csv(os.path.join(results_dir, "detailed_results_real_data.csv"), index=False)
df_summary.to_csv(os.path.join(results_dir, "summary_results_real_data.csv"))

# Save real data information
real_data_info = {
    "data_file": data_file,
    "num_real_voters": real_votes.shape[0],
    "num_metrics": len(metric_ids),
    "metric_ids": metric_ids,
    "voter_ids": voter_ids[:10],  # Save first 10 voter IDs as sample
    "elicitation_method": "cumulative",
    "original_vote_stats": {
        "mean": np.mean(real_votes_raw),
        "std": np.std(real_votes_raw),
        "min": np.min(real_votes_raw),
        "max": np.max(real_votes_raw)
    },
    "row_sums_check": {
        "all_sum_to_1": np.allclose(np.sum(real_votes, axis=1), 1.0),
        "mean_row_sum": np.mean(np.sum(real_votes, axis=1)),
        "std_row_sum": np.std(np.sum(real_votes, axis=1))
    }
}

# Save metadata about the experiment
sim_params = {
    "metrics": metric_ids,
    "num_voters": real_votes.shape[0],
    "num_projects": config.num_projects,
    "budget": config.bribery_budget,
    "cloning_budget": config.cloning_budget,
    "deletion_budget": config.deletion_budget,
    "using_real_data": True,
    "elicitation_method": elicitation_method.value,
    "real_data_info": real_data_info
}

save_simulation_results(results_dir, sim_params, real_votes, value_matrix, ideal_scores, df_results)

# Only create visualizations if we have valid results
if not df_summary.empty:
    visualize_combinations_experiment_results(results_dir, df_summary, df_results_clean)
    print("\nExperiment completed! Results with real voting data have been saved and visualized.")
    print(f"\nSummary of results:")
    print(df_summary)
else:
    print("\nExperiment completed but no valid results were obtained. Check the error logs above.")

print(f"\nResults saved to: {results_dir}")
print(f"Real voting data used: {data_file}")
print(f"Number of real voters: {real_votes.shape[0]}")
print(f"Number of metrics: {len(metric_ids)}")
print(f"Metrics: {metric_ids}")
print(f"Elicitation method: {elicitation_method.value} (real data format)") 