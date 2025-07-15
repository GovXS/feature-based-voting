import pandas as pd
import numpy as np
import os
from typing import Tuple, List

def load_mechanical_turk_data(file_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and format the mechanical turk voting data from Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        votes: Array of shape (num_voters, num_metrics) - voting matrix
        voter_ids: List of voter IDs 
        metric_ids: List of metric IDs
    """
    # Read the CSV-like Excel file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get metric IDs
    header_line = lines[0].strip().strip('"')
    parts = header_line.split(',')
    metric_ids = [part.strip() for part in parts[1:]]  # Skip first column which is "Metric ID/Voter ID"
    
    # Parse data rows
    voter_ids = []
    votes_list = []
    
    for line in lines[1:]:
        line = line.strip().strip('"')
        if not line:
            continue
            
        parts = line.split(',')
        voter_id = parts[0].strip()
        vote_values = [float(part.strip()) for part in parts[1:]]
        
        voter_ids.append(voter_id)
        votes_list.append(vote_values)
    
    # Convert to numpy array
    votes = np.array(votes_list)
    
    print(f"Loaded voting data: {votes.shape[0]} voters, {votes.shape[1]} metrics")
    print(f"Voter IDs: {voter_ids[:5]}... (showing first 5)")
    print(f"Metric IDs: {metric_ids}")
    print(f"Vote matrix shape: {votes.shape}")
    print(f"Sample votes (first voter): {votes[0]}")
    
    return votes, voter_ids, metric_ids

def validate_and_normalize_votes(votes: np.ndarray, elicitation_method: str) -> np.ndarray:
    """
    Validate and normalize votes according to elicitation method constraints.
    
    Args:
        votes: Raw voting matrix
        elicitation_method: The elicitation method being used
        
    Returns:
        Normalized voting matrix
    """
    votes_normalized = votes.copy()
    
    if elicitation_method == "cumulative":
        # For cumulative voting, each row should sum to 1
        row_sums = np.sum(votes_normalized, axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        votes_normalized = votes_normalized / row_sums[:, np.newaxis]
        
    elif elicitation_method == "fractional":
        # For fractional voting, values should be between 0 and 1
        votes_normalized = np.clip(votes_normalized, 0, 1)
        
    elif elicitation_method == "approval":
        # For approval voting, convert to binary (0 or 1)
        # Use threshold of 0.5 or mean as cutoff
        threshold = np.mean(votes_normalized)
        votes_normalized = (votes_normalized > threshold).astype(float)
        
    elif elicitation_method == "plurality":
        # For plurality voting, only highest value in each row should be 1
        votes_normalized = np.zeros_like(votes_normalized)
        max_indices = np.argmax(votes, axis=1)
        votes_normalized[np.arange(len(votes_normalized)), max_indices] = 1
    
    print(f"Normalized votes for {elicitation_method} method")
    print(f"Sample normalized votes (first voter): {votes_normalized[0]}")
    print(f"Row sums (first 5 voters): {np.sum(votes_normalized[:5], axis=1)}")
    
    return votes_normalized

def get_available_data_files() -> List[str]:
    """Get list of available mechanical turk data files."""
    data_dir = "data"
    files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("worldwide_mechanical-turk_utilities") and filename.endswith(".xls"):
            files.append(os.path.join(data_dir, filename))
    return files 