from enum import Enum
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm

class ElicitationMethod(Enum):
    FRACTIONAL = "fractional"  # Each vote between 0 and 1
    CUMULATIVE = "cumulative"  # Sum of votes equals 1
    APPROVAL = "approval"      # Binary (0 or 1)
    PLURALITY = "plurality"    # Single 1, others 0

class VotingSimulator:
    def __init__(self,
                 num_voters: int,
                 metrics: List[str],
                 elicitation_method: ElicitationMethod,
                 alpha: float = 1.0):
        self.num_voters = num_voters
        self.metrics = metrics
        self.elicitation_method = elicitation_method
        self.alpha = alpha  # Mallows model parameter

    def _generate_base_vote(self) -> np.ndarray:
        """Generate a base vote according to the elicitation method"""
        if self.elicitation_method == ElicitationMethod.FRACTIONAL:
            return np.random.uniform(0, 1, len(self.metrics))
            
        elif self.elicitation_method == ElicitationMethod.CUMULATIVE:
            vote = np.random.uniform(0, 1, len(self.metrics))
            return vote / vote.sum()
            
        elif self.elicitation_method == ElicitationMethod.APPROVAL:
            return np.random.choice([0, 1], size=len(self.metrics))
            
        elif self.elicitation_method == ElicitationMethod.PLURALITY:
            vote = np.zeros(len(self.metrics))
            vote[np.random.randint(len(self.metrics))] = 1
            return vote
            
        else:
            raise ValueError(f"Unknown elicitation method: {self.elicitation_method}")

    def _apply_mallows(self, base_vote: np.ndarray) -> np.ndarray:
        """
        Apply Mallows model to generate votes with noise
        Args:
            base_vote: The base vote to perturb
        Returns:
            Perturbed vote that respects the elicitation method constraints
        """
        noise = norm.rvs(scale=1/self.alpha, size=len(self.metrics))
        perturbed = base_vote + noise
        
        # Ensure votes respect the elicitation method constraints
        if self.elicitation_method == ElicitationMethod.FRACTIONAL:
            perturbed = np.clip(perturbed, 0, 1)
            
        elif self.elicitation_method == ElicitationMethod.CUMULATIVE:
            perturbed = np.clip(perturbed, 0, 1)
            perturbed = perturbed / perturbed.sum()
            
        elif self.elicitation_method == ElicitationMethod.APPROVAL:
            perturbed = np.where(perturbed > 0.5, 1, 0)
            
        elif self.elicitation_method == ElicitationMethod.PLURALITY:
            perturbed = np.zeros_like(perturbed)
            perturbed[np.argmax(perturbed)] = 1
            
        return perturbed

    def generate_votes(self) -> np.ndarray:
        """Generate votes using Mallows model"""
        base_vote = self._generate_base_vote()
        votes = np.array([self._apply_mallows(base_vote) for _ in range(self.num_voters)])
        return votes

    def aggregate_votes(self, votes: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Aggregate votes using specified method
        Args:
            votes: Array of shape (num_voters, num_metrics)
            method: Aggregation method ("mean" or "median")
        Returns:
            Aggregated weights of shape (num_metrics,)
        """
        if method == "mean":
            return np.mean(votes, axis=0)
        elif method == "median":
            return np.median(votes, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def compute_scores(self, 
                     value_matrix: np.ndarray, 
                     weights: np.ndarray) -> np.ndarray:
        """
        Compute project scores using value matrix and weights
        Args:
            value_matrix: Array of shape (num_projects, num_metrics)
            weights: Array of shape (num_metrics,)
        Returns:
            Project scores of shape (num_projects,)
        """
        return value_matrix @ weights

# Example usage
if __name__ == "__main__":
    metrics = ["daily_users", "transaction_volume", "unique_wallets"]
    num_voters = 100
    num_projects = 5
    
    # Test different elicitation methods
    for method in ElicitationMethod:
        print(f"\nTesting {method.value} elicitation:")
        
        simulator = VotingSimulator(
            num_voters=num_voters,
            metrics=metrics,
            elicitation_method=method,
            alpha=1.0
        )
        
        # Generate votes
        votes = simulator.generate_votes()
        print("Sample votes:\n", votes[:3])
        
        # Aggregate votes
        weights = simulator.aggregate_votes(votes, method="mean")
        print("Aggregated weights:", weights)
        
        # Generate random value matrix
        value_matrix = np.random.uniform(0, 1, size=(num_projects, len(metrics)))
        
        # Compute project scores
        scores = simulator.compute_scores(value_matrix, weights)
        print("Project scores:", scores) 