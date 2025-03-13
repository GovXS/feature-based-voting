from simulations.bribery_simulation import run_bribery_simulation
from simulations.manipulation_simulation import run_manipulation_simulation
from simulations.deletion_simulation import run_deletion_simulation
from simulations.cloning_simulation import run_cloning_simulation

if __name__ == "__main__":
    print("Running Bribery Simulation...")
    run_bribery_simulation()
    
    print("Running Manipulation Simulation...")
    #run_manipulation_simulation()
    
    print("Running Deletion Simulation...")
    run_deletion_simulation()
    
    print("Running Cloning Simulation...")
    run_cloning_simulation()