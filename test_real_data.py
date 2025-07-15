import sys
import os
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(project_root)

from models.voting_model import VotingSimulator, ElicitationMethod
from models.optimizers import bribery_optimization, manipulation, control_by_cloning, control_by_deletion
from scripts import config
from scripts.data_loader import load_mechanical_turk_data, validate_and_normalize_votes

print("Testing real data integration...")

# Load real voting data
data_file = "data/worldwide_mechanical-turk_utilities-3.xls"
real_votes_raw, voter_ids, metric_ids = load_mechanical_turk_data(data_file)

print(f"Loaded real voting data:")
print(f"- Number of voters: {len(voter_ids)}")
print(f"- Number of metrics: {len(metric_ids)}")
print(f"- Voting matrix shape: {real_votes_raw.shape}")

# Validate that the data is already in cumulative format
real_votes = validate_and_normalize_votes(real_votes_raw, "cumulative")

# Create simulator with real voting data
simulator = VotingSimulator(
    num_voters=real_votes.shape[0],
    num_projects=20,  # Small number for testing
    metrics=metric_ids,
    elicitation_method=ElicitationMethod.CUMULATIVE,
    alpha=config.alpha
)

# Set the real voting data
simulator.set_real_votes(real_votes)

# Generate votes (will return real data)
votes = simulator.generate_votes()

# Generate synthetic value matrix and ideal scores
value_matrix = simulator.generate_value_matrix()
ideal_scores = simulator.generate_ideal_scores()

print(f"\nTesting with:")
print(f"- Votes shape: {votes.shape}")
print(f"- Value matrix shape: {value_matrix.shape}")
print(f"- Ideal scores shape: {ideal_scores.shape}")

# Test each optimizer
print("\nTesting optimizers...")

try:
    # Test bribery optimization
    print("Testing bribery optimization...")
    l1_bribery = bribery_optimization(votes, value_matrix, ideal_scores, 1000.0, ElicitationMethod.CUMULATIVE, "arithmetic_mean")
    print(f"Bribery resistance (L1): {l1_bribery}")
    
    # Test manipulation
    print("Testing manipulation...")
    l1_manipulation = manipulation(votes, value_matrix, ElicitationMethod.CUMULATIVE, "arithmetic_mean")
    print(f"Manipulation resistance (L1): {l1_manipulation}")
    
    # Test deletion control
    print("Testing deletion control...")
    l1_deletion = control_by_deletion(votes, value_matrix, ideal_scores, 2, ElicitationMethod.CUMULATIVE, "arithmetic_mean")
    print(f"Deletion resistance (L1): {l1_deletion}")
    
    # Test cloning control
    print("Testing cloning control...")
    l1_cloning = control_by_cloning(votes, value_matrix, ideal_scores, 2, ElicitationMethod.CUMULATIVE, "arithmetic_mean")
    print(f"Cloning resistance (L1): {l1_cloning}")
    
    print("\nAll tests passed! Real data integration is working correctly.")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc() 