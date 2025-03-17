import numpy as np
from models.VotingModel import VotingSimulator, ElicitationMethod
from models.Metric_based_optimization import bribery_optimization

def run_bribery_simulation(
        votes,
        value_matrix,
        ideal_scores,
        elicitation="cumulative", 
        aggregation="arithmetic_mean",
        budget = 10000.0,
        metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"],
        num_voters = 100,
        num_projects = 200,
        
    ):
   

    # # Initialize simulator
    # simulator = VotingSimulator(
    #     num_voters=num_voters,
    #     metrics=metrics,
    #     elicitation_method=ElicitationMethod.CUMULATIVE,
    #     alpha=1.0
    # )

    # # Generate data
    # votes = simulator.generate_votes()
    # value_matrix = np.random.uniform(0, 1, size=(num_projects, len(metrics)))
    # ideal_scores = np.random.uniform(0, 1, size=num_projects)

    # Run bribery optimization
    min_distance = bribery_optimization(
        votes, value_matrix, ideal_scores, budget,
        elicitation, aggregation
    )

    # Print results
    print("Bribery Optimization Results:")
    print(f"Minimum L1 distance: {min_distance}")
    print("Original votes sample:\n", votes[:3])

    return min_distance


if __name__ == "__main__":
    run_bribery_simulation()
