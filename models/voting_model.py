from enum import Enum
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import combinations
import copy

class ElicitationMethod(Enum):
    FRACTIONAL = "fractional"  # Each vote between 0 and 1
    CUMULATIVE = "cumulative"  # Sum of votes equals 1
    APPROVAL = "approval"      # Binary (0 or 1)
    PLURALITY = "plurality"    # Single 1, others 0

class VotingSimulator:
    def __init__(self,
                 num_voters: int,
                 num_projects: int,
                 metrics: List[str],
                 elicitation_method: ElicitationMethod,
                 alpha: float = 1.0,
                 real_votes: Optional[np.ndarray] = None):
        

        self.num_voters = num_voters
        self.num_projects = num_projects
        self.metrics = metrics
        self.elicitation_method = elicitation_method
        self.alpha = alpha  # Mallows model parameter
        self.real_votes = real_votes  # Optional real voting data

    def set_real_votes(self, votes: np.ndarray):
        """Set real voting data to use instead of synthetic generation."""
        self.real_votes = votes
        self.num_voters = votes.shape[0]
        if votes.shape[1] != len(self.metrics):
            print(f"Warning: Real votes have {votes.shape[1]} metrics, but config specifies {len(self.metrics)}")
            # Update metrics to match real data
            self.metrics = [f"metric_{i}" for i in range(votes.shape[1])]

    def _generate_base_vote(self) -> np.ndarray:
        """Generate a base vote according to the elicitation method"""
        if self.elicitation_method == ElicitationMethod.FRACTIONAL:
            return np.random.uniform(0, 1, len(self.metrics))
            
        elif self.elicitation_method == ElicitationMethod.CUMULATIVE:
            vote = np.random.uniform(0, 1, len(self.metrics))
            # Add small epsilon to avoid division by zero
            vote_sum = vote.sum()
            if vote_sum == 0:
                vote = np.ones(len(self.metrics)) / len(self.metrics)  # Equal distribution if all zeros
            return vote / vote_sum
            
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
            vote_sum = perturbed.sum()
            # Add small epsilon to avoid division by zero
            if vote_sum < 1e-10:  # More robust check for near-zero values
                perturbed = np.ones(len(self.metrics)) / len(self.metrics)
            else:
                perturbed = perturbed / (vote_sum + 1e-10)  # Add epsilon to denominator
            
        elif self.elicitation_method == ElicitationMethod.APPROVAL:
            perturbed = np.where(perturbed > 0.5, 1, 0)
            
        elif self.elicitation_method == ElicitationMethod.PLURALITY:
            perturbed = np.zeros_like(perturbed)
            perturbed[np.argmax(perturbed)] = 1
            
        return perturbed

    def generate_votes(self) -> np.ndarray:
        """Generate votes using real data if available, otherwise use Mallows model"""
        if self.real_votes is not None:
            print(f"Using real voting data with shape {self.real_votes.shape}")
            return self.real_votes
        else:
            print("Using synthetic voting data generation")
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
        # Replace any remaining nan values with 0 before aggregation
        votes = np.nan_to_num(votes, nan=0.0)
        
        if method == "arithmetic_mean":
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
    
    def generate_value_matrix(self):
         value_matrix = np.random.uniform(0, 1, size=(self.num_projects, len(self.metrics)))
         return value_matrix
    
    def normalize_value_matrix(self,value_matrix):
    
        return (value_matrix - value_matrix.min(axis=0)) / (value_matrix.max(axis=0) - value_matrix.min(axis=0))
    
    # Function to introduce strategic modifications to project features
    def modify_project_features(self,value_matrix, modification_ratio=0.1, num_modified=10):
        """Select a subset of projects and modify their feature values."""
        modified_matrix = value_matrix
        selected_projects = np.random.choice(value_matrix.shape[0], num_modified, replace=False)

        for proj in selected_projects:
            # Increase features within a 10% range
            modification = np.random.uniform(1 - modification_ratio, 1 + modification_ratio, size=value_matrix.shape[1])
            modified_matrix[proj, :] *= modification
            modified_matrix[proj, :] = np.clip(modified_matrix[proj, :], 0, 1)  # Keep values within [0,1]

        return modified_matrix, selected_projects
    
    def generate_ideal_scores(self):
        ideal_scores = np.random.uniform(0, 1, size=self.num_projects)

        return ideal_scores




    

        