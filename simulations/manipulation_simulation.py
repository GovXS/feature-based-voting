import numpy as np
from models.VotingModel import VotingSimulator, ElicitationMethod
from models.Metric_based_optimization import manipulation

def run_manipulation_simulation(
        votes,
        value_matrix,
        ideal_scores,
        elicitation="cumulative", 
        aggregation="arithmetic_mean",
        metrics = ["daily_users", "transaction_volume", "unique_wallets"],
    num_voters = 100,
    num_projects = 5,
):
    # Simulation parameters
    

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

    # Run manipulation optimization
    min_distance = manipulation(
        votes, value_matrix,
        elicitation, aggregation
    )

    # Print results
    print("Manipulation Optimization Results:")
    print(f"Minimum L1 distance: {min_distance}")
    print("Original votes sample:\n", votes[:3])

    return min_distance

if __name__ == "__main__":
    run_manipulation_simulation() 