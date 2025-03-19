import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.VotingModel import VotingSimulator, ElicitationMethod

from models.VotingModel import VotingSimulator, ElicitationMethod
import numpy as np
from models.Optimizers import bribery_optimization,manipulation,control_by_cloning, control_by_deletion
from utils.util import save_simulation_results 
from config import experiments_config

if __name__ == "__main__":
    
     # Initialize simulator
    simulator = VotingSimulator(
        num_voters=experiments_config.num_voters,
        num_projects=experiments_config.num_projects,
        metrics=experiments_config.metrics,
        elicitation_method=ElicitationMethod.CUMULATIVE,
        alpha=experiments_config.alpha
    )

    # Generate data
    votes = simulator.generate_votes()
    value_matrix = simulator.generate_value_matrix()
    
    ideal_scores = simulator.generate_ideal_scores()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "main", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("val matrix:",value_matrix)
    

    results = {}
    print("Running Bribery Simulation...")
    results["Bribery"] = min_distance = bribery_optimization(
        votes, value_matrix, ideal_scores, experiments_config.bribery_budget,
        experiments_config.elicitation, experiments_config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")
    
    print("Running Manipulation Simulation...")
    results["Manipulation"]= min_distance = manipulation(
        votes, value_matrix,
        experiments_config.elicitation, experiments_config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")
    


    print("Running Deletion Simulation...")
    results["Deletion"] = min_distance = control_by_deletion(
        votes, value_matrix, ideal_scores, experiments_config.deletion_budget,
        experiments_config.elicitation, experiments_config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")


    print("Running Cloning Simulation...")
    results["Cloning"] =  min_distance = control_by_cloning(
        votes, value_matrix, ideal_scores, experiments_config.cloning_budget,
        experiments_config.elicitation, experiments_config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")

    sim_params = {
        "metrics": experiments_config.metrics,
        "num_voters": experiments_config.num_voters,
        "num_projects": experiments_config.num_projects,
        "budget": experiments_config.bribery_budget,
        "cloning_budget": experiments_config.cloning_budget,
        "deletion_budget": experiments_config.deletion_budget,
       
    }
    save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, results)

    print(f"Simulation results saved to: {results_dir}")