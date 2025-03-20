import os
from datetime import datetime
from models.voting_model import VotingSimulator, ElicitationMethod
from models.Optimizers import bribery_optimization,manipulation,control_by_cloning, control_by_deletion
from utils.util import save_simulation_results
from config import config

metrics =config.metrics
num_voters = config.num_voters
num_projects = config.num_projects
bribery_budget = config.bribery_budget
cloning_budget = config.cloning_budget
deletion_budget = config.deletion_budget

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", "all_combinations", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Run simulations for each combination of elicitation and aggregation
results = {}
for elicitation in [ElicitationMethod.CUMULATIVE, ElicitationMethod.FRACTIONAL, ElicitationMethod.APPROVAL]:
    for aggregation in ["arithmetic_mean", "median"]:
        print(f"Running simulations for {elicitation} elicitation and {aggregation} aggregation...")

        # Initialize simulator
        simulator = VotingSimulator(
            num_voters=num_voters,
            metrics=metrics,
            elicitation_method=elicitation,
            alpha=1.0
        )

        # Generate data
        votes = simulator.generate_votes()
        value_matrix = simulator.generate_value_matrix()
        ideal_scores = simulator.generate_ideal_scores()
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