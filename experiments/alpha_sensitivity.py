import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.voting_model import VotingSimulator, ElicitationMethod
from utils.util import save_simulation_results
from models.Optimizers import bribery_optimization,manipulation,control_by_cloning, control_by_deletion

if __name__ == "__main__":
    # Define metrics and parameters
    metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"]
    num_voters = 3
    num_projects = 4
    bribery_budget = 10000.0
    cloning_budget = 3
    deletion_budget = 3
    alpha_values = [0.1, 0.5,1.0, 2.0, 5.0]  # Test different alpha values

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "alpha_experiment", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Dictionary to store all results
all_results = {}

# Run simulations for each alpha value
for alpha in alpha_values:
    print(f"Running simulations for alpha = {alpha}...")

    # Initialize simulator and generate data
    simulator = VotingSimulator(
        num_voters=num_voters,
        metrics=metrics,
        elicitation_method=ElicitationMethod.CUMULATIVE,
        alpha=alpha
    )
    votes = simulator.generate_votes()
    value_matrix = np.random.uniform(0, 1, size=(num_projects, len(metrics)))
    ideal_scores = np.random.uniform(0, 1, size=num_projects)

    # Normalize value matrix
    value_matrix = simulator.normalize_value_matrix(value_matrix)

    # Create subdirectory for this alpha value
    alpha_dir = os.path.join(results_dir, f"alpha_{alpha}")
    os.makedirs(alpha_dir, exist_ok=True)

    # Run simulations for each combination of elicitation and aggregation
    results = {}
    for elicitation in [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, ElicitationMethod.APPROVAL]:
        for aggregation in ["arithmetic_mean", "geometric_mean"]:
            print(f"  Running {elicitation.value} elicitation and {aggregation} aggregation...")

            # Update simulator with current elicitation method
            simulator.elicitation_method = elicitation

            print(f"Running simulations for {elicitation} elicitation and {aggregation} aggregation...")

            # Update simulator with current elicitation method
            simulator.elicitation_method = elicitation

            # Run bribery simulation
            results[f"Bribery_{elicitation.value}_{aggregation}"] = bribery_optimization(
                                                votes, value_matrix, ideal_scores, bribery_budget,
                                                elicitation, aggregation
                                            )

            
    
   

            # Run manipulation simulation
            results[f"Manipulation_{elicitation.value}_{aggregation}"] = min_distance = manipulation(
                                                                                    votes, value_matrix,
                                                                                    elicitation, aggregation
                                                                                    )
            # Run deletion simulation
            results[f"Deletion_{elicitation.value}_{aggregation}"] = control_by_deletion(
                                    votes, value_matrix, ideal_scores, deletion_budget,
                                    elicitation, aggregation
                                )
            # Run cloning simulation
            results[f"Cloning_{elicitation.value}_{aggregation}"] = control_by_cloning(
                                    votes, value_matrix, ideal_scores, cloning_budget,
                                    elicitation, aggregation
            )
    # Save simulation results for this alpha value
    sim_params = {
        "metrics": metrics,
        "num_voters": num_voters,
        "num_projects": num_projects,
        "budget": bribery_budget,
        "cloning_budget": cloning_budget,
        "deletion_budget": deletion_budget,
        "alpha": alpha
    }
    save_simulation_results(alpha_dir, sim_params, votes, value_matrix, ideal_scores, results)

    # Store results for this alpha value in the consolidated dictionary
    all_results[f"Alpha_{alpha}"] = results

# Save all results in a single file in the main results directory
results_df = pd.DataFrame.from_dict(all_results, orient="index")
results_df.to_csv(os.path.join(results_dir, "all_results.csv"))

print(f"Alpha experiment results saved to: {results_dir}") 