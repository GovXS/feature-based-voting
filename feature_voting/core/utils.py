import numpy as np
from typing import List

def generate_clone_combinations(num_metrics: int, max_clones: int) -> List[List[int]]:
    """Generate all possible cloning combinations"""
    if num_metrics == 1:
        return [[i] for i in range(1, max_clones + 1)]
    else:
        return [
            [i] + rest
            for i in range(1, max_clones + 1)
            for rest in generate_clone_combinations(num_metrics - 1, max_clones)
        ]

def expand_votes(votes: np.ndarray, clone_counts: List[int], elicitation_method) -> np.ndarray:
    """Expand votes according to cloning counts"""
    expanded_votes = []
    for voter in votes:
        expanded_voter = []
        for i, count in enumerate(clone_counts):
            if elicitation_method == ElicitationMethod.CUMULATIVE:
                expanded_voter.extend([voter[i] / count] * count)
            else:
                expanded_voter.extend([voter[i]] * count)
        expanded_votes.append(expanded_voter)
    return np.array(expanded_votes)

def expand_value_matrix(value_matrix: np.ndarray, clone_counts: List[int]) -> np.ndarray:
    """Expand value matrix according to cloning counts"""
    expanded_matrix = []
    for project in value_matrix:
        expanded_project = []
        for i, count in enumerate(clone_counts):
            expanded_project.extend([project[i]] * count)
        expanded_matrix.append(expanded_project)
    return np.array(expanded_matrix) 