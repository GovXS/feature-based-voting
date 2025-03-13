from simulations.bribery_simulation import run_bribery_simulation
from simulations.manipulation_simulation import run_manipulation_simulation
from simulations.deletion_simulation import run_deletion_simulation
from simulations.cloning_simulation import run_cloning_simulation

if __name__ == "__main__":
    metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"]
    num_voters = 100
    num_projects = 200
    budget = 10000.0
    
    print("Running Bribery Simulation...")
    run_bribery_simulation(metrics,num_voters,num_projects,budget)
    
    print("Running Manipulation Simulation...")
    run_manipulation_simulation(metrics,num_voters,num_projects)
    
    print("Running Deletion Simulation...")
    run_deletion_simulation(metrics,num_voters,num_projects)
    
    print("Running Cloning Simulation...")
    run_cloning_simulation(metrics,num_voters,num_projects)