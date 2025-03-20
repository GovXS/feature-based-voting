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

from config import config

elicitation_methods = [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, ElicitationMethod.APPROVAL]
aggregation_methods = ["arithmetic_mean", "median"]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "strategic_voting", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Store results
strategic_results = []


# Run simulations
for elicitation in elicitation_methods:
    print(f"\nRunning simulations for {elicitation.value} elicitation...\n")

    for instance in range(config.num_instances):
        print(f"Generating instance {instance + 1}/{config.num_instances}...")

        # Step 1: Generate voting data
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

        # Step 2: Compute original project scores for comparison
        for aggregation in aggregation_methods:
            print(f"  Testing {aggregation} aggregation...")

            # Compute initial project scores
            imp_agg = simulator.aggregate_votes(votes,aggregation)
            original_scores = simulator.compute_scores(value_matrix, imp_agg)
            original_ranking = np.argsort(original_scores)[::-1]  # Highest score first

            # Step 3: Modify strategic project values
            modified_value_matrix, modified_projects = simulator.modify_project_features(value_matrix)

            # Compute new project scores after strategic modifications
            modified_scores = simulator.compute_scores(modified_value_matrix, imp_agg)
            modified_ranking = np.argsort(modified_scores)[::-1]

            # Step 4: Measure impact of strategic manipulation
            score_change = np.sum(np.abs(modified_scores - original_scores))
            rank_change = np.sum(original_ranking != modified_ranking)

            # Store results
            strategic_results.append({
                "elicitation": elicitation.value,
                "aggregation": aggregation,
                "score_change": score_change,
                "rank_change": rank_change,
                "num_modified_projects": len(modified_projects)
            })

# Convert results to DataFrame and compute averages
df_results = pd.DataFrame(strategic_results)
df_summary = df_results.groupby(["elicitation", "aggregation"]).mean()

# Save results
df_results.to_csv(os.path.join(results_dir, "detailed_results.csv"), index=False)
df_summary.to_csv(os.path.join(results_dir, "summary_results.csv"))


print("\nExperiment completed! Summary results have been saved and displayed.")
