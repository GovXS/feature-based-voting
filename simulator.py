from enum import Enum
import os
import numpy as np
from typing import Dict, List
from scipy import stats
import pandas as pd
from datetime import datetime

class AggregationMethod(Enum):
    ARITHMETIC_MEAN = "arithmetic_mean"
    MEDIAN = "median"
    GEOMETRIC_MEAN = "geometric_mean"
    QUADRATIC = "quadratic"

class VotingSimulator:
    def __init__(self, 
                 total_funds: float = 8_000_000,  # Total OP tokens
                 num_voters: int = 200,
                 num_projects: int = 5,
                 metrics: List[str] = None,
                 aggregation_method: AggregationMethod = AggregationMethod.ARITHMETIC_MEAN):
        """
        Initialize the voting simulator
        
        Args:
            total_funds: Total amount of tokens available for allocation
            num_voters: Number of voters participating
            metrics: List of metric names to track. If None, uses default metrics
            aggregation_method: Method to aggregate votes
        """
        self.metrics = metrics if metrics is not None else []
        self.total_funds = total_funds
        self.num_voters = num_voters
        self.metric_weights = None
        self.num_projects = num_projects
        self.aggregation_method = aggregation_method
        
    def _aggregate_votes(self, all_votes: np.ndarray) -> np.ndarray:
        """
        Aggregate votes using the specified method
        
        Args:
            all_votes: Array of shape (num_voters, num_metrics) containing all votes
        Returns:
            Array of shape (num_metrics,) containing aggregated votes
        """
        if self.aggregation_method == AggregationMethod.ARITHMETIC_MEAN:
            return np.mean(all_votes, axis=0)
            
        elif self.aggregation_method == AggregationMethod.MEDIAN:
            return np.median(all_votes, axis=0)
            
        elif self.aggregation_method == AggregationMethod.GEOMETRIC_MEAN:
            # Add small epsilon to avoid zero values
            return stats.gmean(all_votes + 1e-10, axis=0)
            
        elif self.aggregation_method == AggregationMethod.QUADRATIC:
            # Square the votes before averaging, then take square root
            squared_votes = np.square(all_votes)
            mean_squared = np.mean(squared_votes, axis=0)
            return np.sqrt(mean_squared)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

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
            
        all_votes = np.array(all_votes)
        final_votes = self._aggregate_votes(all_votes)
        
        # Normalize to ensure sum equals total_funds
        final_votes = (final_votes / final_votes.sum()) * self.total_funds
        self.metric_weights = final_votes / self.total_funds
        
        return dict(zip(self.metrics, self.metric_weights))
    
    def allocate_funds(self, projects_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Allocate funds to projects based on their metrics and weights
        
        Args:
            projects: Dict of project_name -> {metric_name: metric_value}
        """
        if self.metric_weights is None:
            raise ValueError("Must run simulate_voting() first")
            
        project_scores = {}
        
        # Calculate weighted score for each project
        for project_name, metrics in projects_metrics.items():
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

    def generate_projects_metrics(self, 
                               metric_ranges: Dict[str, tuple]) -> Dict[str, Dict[str, float]]:
        """
        Generate random projects with metrics within specified ranges
        
        Args:
            num_projects: Number of projects to generate
            metric_ranges: Dict of metric_name -> (min_value, max_value)
        Returns:
            Dict of project_name -> {metric_name: metric_value}
        """
        projects = {}
        
        for i in range(self.num_projects ):
            project_metrics = {}
            for metric, (min_val, max_val) in metric_ranges.items():
                # Generate random value within range
                value = np.random.uniform(min_val, max_val)
                # Round to integer if the range is large enough
                if max_val - min_val > 1:
                    value = round(value)
                project_metrics[metric] = value
            
            projects[f"Project {chr(65 + i)}"] = project_metrics
            
        return projects

# Example usage
if __name__ == "__main__":
    # Define metrics to track
    metrics = [
        "daily_users",
        "transaction_volume",
        "unique_wallets",
        "tvl",
        "developer_activity"
    ]
    num_projects = 100
    num_voters = 300
    total_funds = 10_000_000
    
    # Test different aggregation methods
    for method in AggregationMethod:
        print(f"\nTesting {method.value} aggregation:")
        
        simulator = VotingSimulator(
            total_funds=total_funds,
            num_voters=num_voters,
            metrics=metrics,
            num_projects=num_projects,
            aggregation_method=method
        )
        
        # Simulate voting to determine metric weights
        weights = simulator.simulate_voting()
        print("Metric Weights:", weights)
        
        # Define metric ranges for random project generation
        metric_ranges = {
            "daily_users": (500, 2000),
            "transaction_volume": (100000, 1000000),
            "unique_wallets": (400, 1500),
            "tvl": (1000000, 5000000),
            "developer_activity": (10, 100)
        }

        # Create data directory if it doesn't exist
        os.makedirs('data/simulation_data/projects_metrics', exist_ok=True)
        os.makedirs('data/simulation_data/fund_allocation', exist_ok=True)
        
        # Generate random projects
        projects_metrics = simulator.generate_projects_metrics(metric_ranges=metric_ranges)
        
        # Allocate funds based on metrics and weights
        allocations = simulator.allocate_funds(projects_metrics)

        # Save results to CSV files
        projects_df = pd.DataFrame.from_dict(projects_metrics, orient='index')
        projects_df.index.name = 'project'
        projects_df.to_csv(f'data/simulation_data/projects_metrics/projects_metrics_{method.value}.csv')
        
        allocations_df = pd.DataFrame.from_dict(allocations, 
                                              orient='index', 
                                              columns=['allocation'])
        allocations_df.index.name = 'project'
        allocations_df.to_csv(f'data/simulation_data/fund_allocation/fund_allocation_{method.value}.csv')

        print("\nFund Allocations (OP tokens):")
        for project, amount in allocations.items():
            print(f"{project}: {amount:,.0f}")
