import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.VotingModel import VotingSimulator, ElicitationMethod
from simulations.bribery_simulation import run_bribery_simulation
from simulations.manipulation_simulation import run_manipulation_simulation
from simulations.deletion_simulation import run_deletion_simulation
from simulations.cloning_simulation import run_cloning_simulation
from models.VotingModel import VotingSimulator, ElicitationMethod
import numpy as np

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
    metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"]

    num_voters = 3
    num_projects = 4
    bribery_budget = 10000.0
    cloning_budget = 3
    deletion_budget = 3
    elicitation =ElicitationMethod.CUMULATIVE,
    aggregation = "arithmetic_mean",

     # Initialize simulator
    simulator = VotingSimulator(
        num_voters=num_voters,
        metrics=metrics,
        elicitation_method=ElicitationMethod.CUMULATIVE,
        alpha=1.0
    )

    # Generate data
    votes = simulator.generate_votes()
    value_matrix = np.random.uniform(0, 1, size=(num_projects, len(metrics)))

    
    ideal_scores = np.random.uniform(0, 1, size=num_projects)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "main", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("val matrix:",value_matrix)


    

    results = {}
    print("Running Bribery Simulation...")
    results["Bribery"] = run_bribery_simulation(votes, value_matrix, ideal_scores,  elicitation, aggregation,bribery_budget,metrics,num_voters,num_projects)
    
    print("Running Manipulation Simulation...")
    results["Manipulation"]= run_manipulation_simulation(votes, value_matrix, ideal_scores,elicitation, aggregation,metrics,num_voters,num_projects)
    
    print("Running Deletion Simulation...")
    results["Deletion"] = run_deletion_simulation(votes, value_matrix, ideal_scores,elicitation, aggregation,deletion_budget,metrics,num_voters,num_projects)
    
    print("Running Cloning Simulation...")
    results["Cloning"] = run_cloning_simulation(votes, value_matrix, ideal_scores,elicitation, aggregation,cloning_budget, metrics,num_voters,num_projects)

    sim_params = {
        "metrics": metrics,
        "num_voters": num_voters,
        "num_projects": num_projects,
        "budget": bribery_budget,
        "cloning_budget": cloning_budget,
        "deletion_budget": deletion_budget,
       
    }
    save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, results)

    print(f"Simulation results saved to: {results_dir}")