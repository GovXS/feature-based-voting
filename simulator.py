import numpy as np
from typing import Dict, List

class VotingSimulator:
    def __init__(self, 
                 total_funds: float = 8_000_000,  # Total OP tokens
                 num_voters: int = 200,
                 metrics: List[str] = None):
        
        if metrics is None:
            self.metrics = [
                "daily_users",
                "transaction_volume",
                "unique_wallets"
            ]
        else:
            self.metrics = metrics
            
        self.total_funds = total_funds
        self.num_voters = num_voters
        self.metric_weights = None
        
    def simulate_voting(self) -> Dict[str, float]:
        """Simulate voters allocating funds to different metrics"""
        all_votes = []
        
        # Each voter allocates their share of tokens
        for _ in range(self.num_voters):
            # Generate random weights that sum to 1
            votes = np.random.dirichlet(np.ones(len(self.metrics)))
            # Scale votes by total funds
            votes = votes * self.total_funds
            all_votes.append(votes)
            
        # Average all votes to get final metric weights
        final_votes = np.mean(all_votes, axis=0)
        
        # Convert to percentage weights
        self.metric_weights = final_votes / self.total_funds
        
        return dict(zip(self.metrics, self.metric_weights))
    
    def allocate_funds(self, projects: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Allocate funds to projects based on their metrics and weights
        
        Args:
            projects: Dict of project_name -> {metric_name: metric_value}
        """
        if self.metric_weights is None:
            raise ValueError("Must run simulate_voting() first")
            
        project_scores = {}
        
        # Calculate weighted score for each project
        for project_name, metrics in projects.items():
            score = 0
            for metric, value in metrics.items():
                weight_index = self.metrics.index(metric)
                score += value * self.metric_weights[weight_index]
            project_scores[project_name] = score
            
        # Normalize scores to get fund allocation
        total_score = sum(project_scores.values())
        fund_allocation = {
            project: (score / total_score) * self.total_funds 
            for project, score in project_scores.items()
        }
        
        return fund_allocation

# Example usage
if __name__ == "__main__":
    # Initialize simulator
    simulator = VotingSimulator()
    
    # Simulate voting to determine metric weights
    weights = simulator.simulate_voting()
    print("Metric Weights:", weights)
    
    # Example projects with their metrics
    projects = {
        "Project A": {
            "daily_users": 1000,
            "transaction_volume": 500000,
            "unique_wallets": 800
        },
        "Project B": {
            "daily_users": 800,
            "transaction_volume": 700000,
            "unique_wallets": 600
        },
        "Project C": {
            "daily_users": 1200,
            "transaction_volume": 300000,
            "unique_wallets": 1000
        }
    }
    
    # Allocate funds based on metrics and weights
    allocations = simulator.allocate_funds(projects)
    print("\nFund Allocations (OP tokens):")
    for project, amount in allocations.items():
        print(f"{project}: {amount:,.0f}")
