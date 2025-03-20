import sys
import os

# Get the project root directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.voting_model import VotingSimulator, ElicitationMethod
from models.optimizers import bribery_optimization, manipulation, control_by_cloning, control_by_deletion
from config import config
from utils.util import save_simulation_results


elicitation_methods = [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, ElicitationMethod.APPROVAL, ElicitationMethod.PLURALITY]
aggregation_methods = ["arithmetic_mean", "median"]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "intsances", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Storage for final results
resistance_results = []

# Run simulations
for elicitation in elicitation_methods:
    print(f"\nRunning simulations for {elicitation.value} elicitation...\n")

    for instance in range(config.num_instances):
        print(f"Generating instance {instance + 1}/{config.num_instances}...")

        # Step 1: Generate one instance of votes, value matrix, and ideal scores
        simulator = VotingSimulator(
            num_voters=config.num_voters,
            num_projects=config.num_projects,
            metrics=config.metrics,
            elicitation_method=elicitation,
            alpha=config.alpha
        )
        votes = simulator.generate_votes()
        value_matrix = simulator.generate_value_matrix()
        ideal_scores = simulator.generate_ideal_scores()

        # Step 2: Evaluate all problems for each aggregation method
        for aggregation in aggregation_methods:
            print(f"  Testing {aggregation} aggregation...")

            # Bribery Optimization
            l1_bribery = bribery_optimization(votes, value_matrix, ideal_scores, config.bribery_budget, elicitation, aggregation)

            # Manipulation
            l1_manipulation = manipulation(votes, value_matrix, elicitation, aggregation)

            # Feature Deletion Control
            l1_deletion = control_by_deletion(votes, value_matrix, ideal_scores, config.deletion_budget, elicitation, aggregation)

            # Feature Cloning Control
            l1_cloning = control_by_cloning(votes, value_matrix, ideal_scores, config.cloning_budget, elicitation, aggregation)

            # Store results
            resistance_results.append({
                "elicitation": elicitation.value,
                "aggregation": aggregation,
                "bribery_resistance": l1_bribery,
                "manipulation_resistance": l1_manipulation,
                "deletion_resistance": l1_deletion,
                "cloning_resistance": l1_cloning
            })
            print(resistance_results)

# Convert results to DataFrame and compute average L1 distances
df_results = pd.DataFrame(resistance_results)
df_summary = df_results.groupby(["elicitation", "aggregation"]).mean()

# Save detailed and summary results
df_results.to_csv(os.path.join(results_dir, "detailed_results.csv"), index=False)
df_summary.to_csv(os.path.join(results_dir, "summary_results.csv"))

sim_params = {
    "metrics": config.metrics,
    "num_voters": config.num_voters,
    "num_projects": config.num_projects,
    "budget": config.bribery_budget,
    "cloning_budget": config.cloning_budget,
    "deletion_budget": config.deletion_budget,
    
}
save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, df_results)

print(df_summary)

print("\nExperiment completed! Summary results have been saved and displayed.")
