import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.voting_model import VotingSimulator, ElicitationMethod
from scripts.util import save_simulation_results
from models.optimizers import bribery_optimization,manipulation,control_by_cloning, control_by_deletion
from scripts import config

if __name__ == "__main__":
   
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
            num_voters=config.num_voters,
            metrics=config.metrics,
            num_projects=config.num_projects,
            elicitation_method=ElicitationMethod.CUMULATIVE,
            alpha=alpha
        )
    # Run simulations for each instance
        instance_results = []
        for instance in range(config.num_instances):
            print(f"  Instance {instance + 1}/{config.num_instances}...")

            # Generate data
            votes = simulator.generate_votes()
            value_matrix = simulator.generate_value_matrix()
            ideal_scores = simulator.generate_ideal_scores()

            # Normalize value matrix
            value_matrix = simulator.normalize_value_matrix(value_matrix)

            # Run simulations for each combination of elicitation and aggregation
            for elicitation in [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, ElicitationMethod.APPROVAL]:
                for aggregation in ["arithmetic_mean", "geometric_mean"]:
                    print(f"    Running {elicitation.value} elicitation and {aggregation} aggregation...")

                    # Bribery Optimization
                    l1_bribery = bribery_optimization(votes, value_matrix, ideal_scores, config.bribery_budget, elicitation, aggregation)

                    # Manipulation
                    l1_manipulation = manipulation(votes, value_matrix, elicitation, aggregation)

                    # Feature Deletion Control
                    l1_deletion = control_by_deletion(votes, value_matrix, ideal_scores, config.deletion_budget, elicitation, aggregation)

                    # Feature Cloning Control
                    l1_cloning = control_by_cloning(votes, value_matrix, ideal_scores, config.cloning_budget, elicitation, aggregation)

                    # Store results for this instance
                    instance_results.append({
                        "alpha": alpha,
                        "instance": instance + 1,
                        "elicitation": elicitation.value,
                        "aggregation": aggregation,
                        "bribery_l1": l1_bribery,
                        "manipulation_l1": l1_manipulation,
                        "deletion_l1": l1_deletion,
                        "cloning_li": l1_cloning
                    })

        # Compute mean L1 distances for this alpha
        df_instance_results = pd.DataFrame(instance_results)
        mean_results = df_instance_results.groupby(["alpha", "elicitation", "aggregation"]).mean().reset_index()

        # Store results for this alpha
        all_results.append(mean_results)

    # Combine all results into a single DataFrame
    df_all_results = pd.concat(all_results)

    # Save results
    df_all_results.to_csv(os.path.join(results_dir, "alpha_sensitivity_results.csv"), index=False)

    sim_params = {
        "metrics": config.metrics,
        "num_voters": config.num_voters,
        "num_projects": config.num_projects,
        "budget": config.bribery_budget,
        "cloning_budget": config.cloning_budget,
        "deletion_budget": config.deletion_budget,
        
    }

    save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores,df_all_results)

    print(f"Alpha sensitivity experiment results saved to: {results_dir}")