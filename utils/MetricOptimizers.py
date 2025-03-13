import numpy as np
import Tuple

def bribery_optimization(self,
                           votes: np.ndarray,
                           value_matrix: np.ndarray,
                           ideal_scores: np.ndarray,
                           budget: float,
                           method: str = "mean") -> Tuple[float, np.ndarray]:
        """
        Find optimal vote shifts to minimize distance to ideal scores
        
        Args:
            votes: Current votes (num_voters x num_metrics)
            value_matrix: Project metrics (num_projects x num_metrics)
            ideal_scores: Desired project scores (num_projects,)
            budget: Maximum total shift allowed
            method: Aggregation method ("mean" or "median")
        Returns:
            Tuple of (min_distance, optimized_votes)
        """
        num_voters, num_metrics = votes.shape
        
        def objective(shifts: np.ndarray) -> float:
            # Reshape shifts and apply to votes
            shifts = shifts.reshape((num_voters, num_metrics))
            new_votes = np.clip(votes + shifts, 0, 1)
            
            # Ensure cumulative constraints if needed
            if self.elicitation_method == ElicitationMethod.CUMULATIVE:
                new_votes = new_votes / new_votes.sum(axis=1, keepdims=True)
            
            # Compute new weights and scores
            weights = self.aggregate_votes(new_votes, method)
            new_scores = self.compute_scores(value_matrix, weights)
            
            # L1 distance to ideal scores
            return np.linalg.norm(new_scores - ideal_scores, ord=1)
        
        # Constraints
        constraints = [
            # Total shift cannot exceed budget
            {'type': 'ineq', 'fun': lambda x: budget - np.sum(np.abs(x))}
        ]
        
        # Bounds for shifts
        bounds = [(-1, 1)] * (num_voters * num_metrics)
        
        # Initial guess (no shift)
        x0 = np.zeros(num_voters * num_metrics)
        
        # Optimize
        result = minimize(objective, x0, 
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        # Reshape optimal shifts
        optimal_shifts = result.x.reshape((num_voters, num_metrics))
        optimal_votes = np.clip(votes + optimal_shifts, 0, 1)
        
        return result.fun, optimal_votes

    def control_by_deletion(self,
                          votes: np.ndarray,
                          value_matrix: np.ndarray,
                          ideal_scores: np.ndarray,
                          max_deletions: int,
                          method: str = "mean") -> Tuple[float, List[int]]:
        """
        Find optimal metric deletion to minimize distance to ideal scores
        
        Args:
            votes: Current votes (num_voters x num_metrics)
            value_matrix: Project metrics (num_projects x num_metrics)
            ideal_scores: Desired project scores (num_projects,)
            max_deletions: Maximum number of metrics to delete
            method: Aggregation method ("mean" or "median")
        Returns:
            Tuple of (min_distance, list of indices to delete)
        """
        num_metrics = votes.shape[1]
        min_distance = float('inf')
        best_deletion = []
        
        # Try all possible combinations of deletions
        for num_del in range(1, max_deletions + 1):
            for del_indices in combinations(range(num_metrics), num_del):
                # Create mask for remaining metrics
                mask = np.ones(num_metrics, dtype=bool)
                mask[list(del_indices)] = False
                
                # Handle cumulative case
                if self.elicitation_method == ElicitationMethod.CUMULATIVE:
                    remaining_votes = votes[:, mask]
                    remaining_votes = remaining_votes / remaining_votes.sum(axis=1, keepdims=True)
                else:
                    remaining_votes = votes[:, mask]
                
                # Compute scores with remaining metrics
                weights = self.aggregate_votes(remaining_votes, method)
                remaining_value_matrix = value_matrix[:, mask]
                scores = self.compute_scores(remaining_value_matrix, weights)
                
                # Compute distance
                distance = np.linalg.norm(scores - ideal_scores, ord=1)
                
                if distance < min_distance:
                    min_distance = distance
                    best_deletion = list(del_indices)
        
        return min_distance, best_deletion

    def control_by_cloning(self,
                         votes: np.ndarray,
                         value_matrix: np.ndarray,
                         ideal_scores: np.ndarray,
                         max_clones: int,
                         method: str = "mean") -> Tuple[float, Dict[int, int]]:
        """
        Find optimal metric cloning to minimize distance to ideal scores
        
        Args:
            votes: Current votes (num_voters x num_metrics)
            value_matrix: Project metrics (num_projects x num_metrics)
            ideal_scores: Desired project scores (num_projects,)
            max_clones: Maximum number of clones to create
            method: Aggregation method ("mean" or "median")
        Returns:
            Tuple of (min_distance, dict of {metric_index: num_clones})
        """
        num_metrics = votes.shape[1]
        min_distance = float('inf')
        best_cloning = {}
        
        # Try all possible cloning combinations
        for clone_counts in self._generate_clone_combinations(num_metrics, max_clones):
            # Create expanded votes and value matrix
            expanded_votes = self._expand_votes(votes, clone_counts)
            expanded_value_matrix = self._expand_value_matrix(value_matrix, clone_counts)
            
            # Compute scores
            weights = self.aggregate_votes(expanded_votes, method)
            scores = self.compute_scores(expanded_value_matrix, weights)
            
            # Compute distance
            distance = np.linalg.norm(scores - ideal_scores, ord=1)
            
            if distance < min_distance:
                min_distance = distance
                best_cloning = {i: count for i, count in enumerate(clone_counts) if count > 1}
        
        return min_distance, best_cloning

    def _generate_clone_combinations(self, num_metrics: int, max_clones: int) -> List[List[int]]:
        """Generate all possible cloning combinations"""
        # This is a recursive generator for all possible cloning counts
        # Each metric can be cloned 1 to max_clones times
        if num_metrics == 1:
            return [[i] for i in range(1, max_clones + 1)]
        else:
            return [
                [i] + rest
                for i in range(1, max_clones + 1)
                for rest in self._generate_clone_combinations(num_metrics - 1, max_clones)
            ]

    def _expand_votes(self, votes: np.ndarray, clone_counts: List[int]) -> np.ndarray:
        """Expand votes according to cloning counts"""
        expanded_votes = []
        for voter in votes:
            expanded_voter = []
            for i, count in enumerate(clone_counts):
                if self.elicitation_method == ElicitationMethod.CUMULATIVE:
                    expanded_voter.extend([voter[i] / count] * count)
                else:
                    expanded_voter.extend([voter[i]] * count)
            expanded_votes.append(expanded_voter)
        return np.array(expanded_votes)

    def _expand_value_matrix(self, value_matrix: np.ndarray, clone_counts: List[int]) -> np.ndarray:
        """Expand value matrix according to cloning counts"""
        expanded_matrix = []
        for project in value_matrix:
            expanded_project = []
            for i, count in enumerate(clone_counts):
                expanded_project.extend([project[i]] * count)
            expanded_matrix.append(expanded_project)
        return np.array(expanded_matrix)

    def manipulation(self,
                    votes: np.ndarray,
                    value_matrix: np.ndarray,
                    ideal_scores: np.ndarray,
                    method: str = "mean") -> Tuple[float, np.ndarray]:
        """
        Find optimal votes for each agent to minimize distance to their ideal scores
        
        Args:
            votes: Current votes (num_voters x num_metrics)
            value_matrix: Project metrics (num_projects x num_metrics)
            ideal_scores: Desired project scores (num_projects,)
            method: Aggregation method ("mean" or "median")
        Returns:
            Tuple of (min_distance, optimized_votes)
        """
        num_voters, num_metrics = votes.shape
        optimized_votes = np.zeros_like(votes)
        
        for i in range(num_voters):
            def objective(vote: np.ndarray) -> float:
                # Create new votes with this voter's optimized vote
                new_votes = np.copy(votes)
                new_votes[i] = vote
                
                # Ensure cumulative constraints if needed
                if self.elicitation_method == ElicitationMethod.CUMULATIVE:
                    new_votes[i] = new_votes[i] / new_votes[i].sum()
                
                # Compute new weights and scores
                weights = self.aggregate_votes(new_votes, method)
                new_scores = self.compute_scores(value_matrix, weights)
                
                # L1 distance to ideal scores
                return np.linalg.norm(new_scores - ideal_scores, ord=1)
            
            # Constraints and bounds based on elicitation method
            if self.elicitation_method == ElicitationMethod.FRACTIONAL:
                bounds = [(0, 1)] * num_metrics
            elif self.elicitation_method == ElicitationMethod.CUMULATIVE:
                bounds = [(0, 1)] * num_metrics
                constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            elif self.elicitation_method == ElicitationMethod.APPROVAL:
                bounds = [(0, 1)] * num_metrics
            elif self.elicitation_method == ElicitationMethod.PLURALITY:
                bounds = [(0, 1)] * num_metrics
            
            # Initial guess (current vote)
            x0 = votes[i]
            
            # Optimize
            if self.elicitation_method == ElicitationMethod.CUMULATIVE:
                result = minimize(objective, x0,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints)
            else:
                result = minimize(objective, x0,
                                method='SLSQP',
                                bounds=bounds)
            
            # Store optimized vote
            optimized_votes[i] = result.x
        
        # Compute final distance with optimized votes
        final_weights = self.aggregate_votes(optimized_votes, method)
        final_scores = self.compute_scores(value_matrix, final_weights)
        min_distance = np.linalg.norm(final_scores - ideal_scores, ord=1)
        
        return min_distance, optimized_votes
