import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.VotingModel import VotingSimulator, ElicitationMethod
from simulations.bribery_simulation import run_bribery_simulation
from simulations.manipulation_simulation import run_manipulation_simulation
from simulations.deletion_simulation import run_deletion_simulation
from simulations.cloning_simulation import run_cloning_simulation

def normalize_value_matrix(value_matrix):
    """Normalize each metric in the value matrix to the range [0, 1]."""
    return (value_matrix - value_matrix.min(axis=0)) / (value_matrix.max(axis=0) - value_matrix.min(axis=0))

def save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, results):
    # Convert sim_params to a format that pd.DataFrame.from_dict can handle
    sim_params_cleaned = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in sim_params.items()}

    # Save simulation parameters
    pd.DataFrame.from_dict(sim_params_cleaned, orient="index", columns=["Value"]).to_csv(os.path.join(results_dir, "sim_params.csv"))

    # Save votes
    pd.DataFrame(votes).to_csv(os.path.join(results_dir, "votes.csv"), index=False)

    # Save value matrix
    pd.DataFrame(value_matrix).to_csv(os.path.join(results_dir, "value_matrix.csv"), index=False)

    # Save ideal scores
    pd.DataFrame(ideal_scores, columns=["ideal_scores"]).to_csv(os.path.join(results_dir, "ideal_scores.csv"), index=False)

    # Save results
    pd.DataFrame.from_dict(results, orient="index", columns=["Minimum L1 Distance"]).to_csv(os.path.join(results_dir, "results.csv"))


if __name__ == "__main__":
    # Define metrics and parameters
    metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"]
    num_voters = 100
    num_projects = 200
    budget = 10000.0
    max_clones = 2
    max_deletions = 1
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Test different alpha values

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "alpha_experiment", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

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
        value_matrix = normalize_value_matrix(value_matrix)

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

                # Run bribery simulation
                results[f"Bribery_{elicitation.value}_{aggregation}"] = run_bribery_simulation(
                    votes, value_matrix, ideal_scores, budget, elicitation.value, aggregation
                )

                # Run manipulation simulation
                results[f"Manipulation_{elicitation.value}_{aggregation}"] = run_manipulation_simulation(
                    votes, value_matrix, ideal_scores, elicitation.value, aggregation
                )

                # Run deletion simulation
                results[f"Deletion_{elicitation.value}_{aggregation}"] = run_deletion_simulation(
                    votes, value_matrix, ideal_scores, max_deletions, elicitation.value, aggregation
                )

                # Run cloning simulation
                results[f"Cloning_{elicitation.value}_{aggregation}"] = run_cloning_simulation(
                    votes, value_matrix, ideal_scores, max_clones, elicitation.value, aggregation
                )

        # Save simulation results for this alpha value
        sim_params = {
            "metrics": metrics,
            "num_voters": num_voters,
            "num_projects": num_projects,
            "budget": budget,
            "max_clones": max_clones,
            "max_deletions": max_deletions,
            "alpha": alpha
        }
        save_simulation_results(alpha_dir, sim_params, votes, value_matrix, ideal_scores, results)

    print(f"Alpha experiment results saved to: {results_dir}") 