# Experiment parameters
from models.voting_model import VotingSimulator, ElicitationMethod

num_instances = 200  # Number of independent instances
num_voters = 100
num_projects = 200
bribery_budget = 10000.0
cloning_budget = 2
deletion_budget = 2
metrics = ["daily_users", "transaction_volume", "unique_wallets", "tvl"]
alpha = 1
aggregation = "arithmetic_mean"
elicitation = ElicitationMethod.CUMULATIVE