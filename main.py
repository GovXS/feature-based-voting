import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.voting_model import VotingSimulator, ElicitationMethod

from models.voting_model import VotingSimulator, ElicitationMethod
import numpy as np
from models.optimizers import bribery_optimization,manipulation,control_by_cloning, control_by_deletion
from scripts.util import save_simulation_results 
from scripts import config

if __name__ == "__main__":
    
     # Initialize simulator
    simulator = VotingSimulator(
        num_voters=config.num_voters,
        num_projects=config.num_projects,
        metrics=config.metrics,
        elicitation_method=ElicitationMethod.CUMULATIVE,
        alpha=config.alpha
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
        votes, value_matrix, ideal_scores, config.bribery_budget,
        config.elicitation, config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")
    
    print("Running Manipulation Simulation...")
    results["Manipulation"]= min_distance = manipulation(
        votes, value_matrix,
        config.elicitation, config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")
    


    print("Running Deletion Simulation...")
    results["Deletion"] = min_distance = control_by_deletion(
        votes, value_matrix, ideal_scores, config.deletion_budget,
        config.elicitation, config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")


    print("Running Cloning Simulation...")
    results["Cloning"] =  min_distance = control_by_cloning(
        votes, value_matrix, ideal_scores, config.cloning_budget,
        config.elicitation, config.aggregation
    )

    # Print results
    print(f"Minimum L1 distance: {min_distance}")

    sim_params = {
        "metrics": config.metrics,
        "num_voters": config.num_voters,
        "num_projects": config.num_projects,
        "budget": config.bribery_budget,
        "cloning_budget": config.cloning_budget,
        "deletion_budget": config.deletion_budget,
       
    }
    save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, results)

    print(f"Simulation results saved to: {results_dir}")