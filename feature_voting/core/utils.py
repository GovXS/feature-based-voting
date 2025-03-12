import cvxpy as cp  # need to install pyscipopt
import numpy as np
from typing import Dict, List, Tuple

class VotingOptimizer:
    @staticmethod
    def bribery_optimization(simulator,votes, value_matrix, scores, budget, elicitation, aggregation):
    
        n, k = votes.shape  # Number of voters and features
        m = value_matrix.shape[0]  # Number of projects
    
        # Decision variables
        z = cp.Variable((n, k), nonneg=True)  # Absolute bribery cost per voter-feature
        score_prime = cp.Variable(m)  # New bribed scores
    
        # Constraints
        constraints = []
    
        # Elicitation methods
        if elicitation == "fractional":
            imp_prime = cp.Variable((n, k), nonneg=True)  # Continuous variable
        elif elicitation == "cumulative":
            imp_prime = cp.Variable((n, k), nonneg=True)  # Continuous variable
            row_sum = np.sum(votes, axis=1)[0]
            for i in range(n):
                constraints.append(cp.sum(imp_prime[i, :]) == row_sum)
        elif elicitation == "approval":
            imp_prime = cp.Variable((n, k), boolean=True)  # Binary variable (0 or 1)
        elif elicitation == "plurality":
            imp_prime = cp.Variable((n, k), boolean=True)  # Binary variable (0 or 1)
            row_sum = np.sum(votes, axis=1)[0]
            for i in range(n):
                constraints.append(cp.sum(imp_prime[i, :]) == row_sum)
    
        # Aggregation: Arithmetic Mean
        if aggregation == "arithmetic_mean":
            for j in range(m):
                constraints.append(score_prime[j] == cp.sum(cp.multiply(cp.sum(imp_prime, axis=0) / n, value_matrix[j, :])))
    
            # Bribery constraints
            for i in range(n):
                for f in range(k):
                    constraints.append(z[i, f] >= imp_prime[i, f] - votes[i, f])
                    constraints.append(z[i, f] >= votes[i, f] - imp_prime[i, f])
    
            constraints.append(cp.sum(z) <= budget)
    
        # Aggregation: Median
        elif aggregation == "median":
            v_prime = cp.Variable(k)  # Bribed feature importance values
            c = cp.Variable(k, nonneg=True)  # Deviation from median
            median_values = np.median(votes, axis=0)  # Compute feature-wise median
    
            # Define constraints for median computation per feature
            for f in range(k):
                if n % 2 == 1:  # Odd number of voters
                    constraints.append(v_prime[f] - median_values[f] <= c[f])
                    constraints.append(median_values[f] - v_prime[f] <= c[f])
                else:  # Even number of voters
                    constraints.append(2 * (v_prime[f] - median_values[f]) <= c[f])
                    constraints.append(2 * (median_values[f] - v_prime[f]) <= c[f])
            
            # Range constraints (for each voter and feature)
            for i in range(n):
                for f in range(k):
                    # Calculate the number of voters with votes above or below the median for feature f
                    count_above = sum(1 for j in range(n) if j != i and votes[j, f] >= median_values[f])
                    count_below = sum(1 for j in range(n) if j != i and votes[j, f] <= median_values[f])
                    
                    # Ensure the constraints are only applied when these counts are >0
                    if count_above > 0:
                        sum_above = cp.sum([votes[j, f] for j in range(n) if j != i and votes[j, f] >= median_values[f]])
                        constraints.append(v_prime[f] * count_above - sum_above <= c[f])
                    if count_below > 0:
                        sum_below = cp.sum([votes[j, f] for j in range(n) if j != i and votes[j, f] <= median_values[f]])
                        constraints.append(sum_below - v_prime[f] * count_below <= c[f])
    
            # Score constraints (only once, not inside the voter loop)
            for j in range(m):
                constraints.append(score_prime[j] == cp.sum(cp.multiply(v_prime, value_matrix[j, :])))
            
            # Budget constraint on total deviation
            constraints.append(cp.sum(c) <= budget)
    
        # L1 distance variable
        l1_distance = cp.norm1(score_prime - scores)
    
        # Objective: Minimize L1 distance ||new_scores - scores||
        objective = cp.Minimize(l1_distance)
    
        # Solve the LP/MILP
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver=cp.SCIP if elicitation == "approval" or elicitation == "plurality" else cp.SCS)  # Use MILP solver if binary
    
        if problem.status in ["infeasible", "unbounded"]:
            print(problem.status)
            return None  # Problem is infeasible or unbounded, return None instead of failing
        
        return l1_distance.value   

    @staticmethod
    def manipulation(simulator,votes, value_matrix, elicitation,aggregation):
        """
        Computes the optimal manipulative vote for agent `agent_index` that minimizes L1 distance 
        between the manipulated score and the ideal score.
    
        Parameters:
        - votes: (n, k) matrix of importance votes.
        - value_matrix: (m, k) project values over features.
        - agent_index: The index of the manipulating agent.
        - method: 'mean' or 'median' aggregation.
    
        Returns:
        - L1 distance between the manipulated score and ideal score.
        - The manipulated vote vector for agent `agent_index`.
        """
        n, k = votes.shape  # Number of voters and features
        m = value_matrix.shape[0]  # Number of projects
        random_agent_index = np.random.choice(n)
        score_prime = cp.Variable(m)  # New manipulated scores
        # Compute the ideal score for the manipulator
        ideal_score = value_matrix @ votes[agent_index, :]
    
        constraints = []
    
        # Elicitation methods
        if elicitation == "fractional":
            imp_prime = cp.Variable(k, nonneg=True)  # Manipulated vote for agent i'
        elif elicitation == "cumulative":
            imp_prime = cp.Variable(k, nonneg=True)  # Continuous variable
            row_sum = np.sum(votes, axis=1)[0]
            constraints.append(cp.sum(imp_prime) == row_sum)
        elif elicitation == "approval":
            imp_prime = cp.Variable(k, boolean=True)  # Binary variable (0 or 1)
        elif elicitation == "plurality":
            imp_prime = cp.Variable((n, k), boolean=True)  # Binary variable (0 or 1)
            row_sum = np.sum(votes, axis=1)[0]
            constraints.append(cp.sum(imp_prime) == row_sum)
    
        if aggregation == "mean":
            # Aggregated importance after manipulation
            imp_agg = (cp.sum(votes, axis=0) - votes[agent_index, :] + imp_prime) / n
    
            # Compute the new project scores
            for j in range(m):
                constraints.append(score_prime[j] == cp.sum(cp.multiply(imp_agg, value_matrix[j, :])))
        
        elif aggregation == "median":
            # Compute the median values excluding the manipulator
            votes_excluding_agent = np.delete(votes, agent_index, axis=0)
            median_values = np.median(votes_excluding_agent, axis=0)
    
            # Constraints ensuring imp_prime is between the relevant values
            for f in range(k):
                constraints.append(imp_prime[f] >= cp.min(votes_excluding_agent[:, f]))  # Stay above lowest
                constraints.append(imp_prime[f] <= cp.max(votes_excluding_agent[:, f]))  # Stay below highest
            
            # If n is odd: manipulated vote should be close to median
            if (n - 1) % 2 == 1:
                c_f = cp.Variable(k, nonneg=True)  # Deviation from median
                for f in range(k):
                    constraints.append(imp_prime[f] - median_values[f] <= c_f[f])
                    constraints.append(median_values[f] - imp_prime[f] <= c_f[f])
            
            # If n is even: minimize deviation from median
            else:
                c_f = cp.Variable(k, nonneg=True)
                for f in range(k):
                    constraints.append(2 * (imp_prime[f] - median_values[f]) <= c_f[f])
                    constraints.append(2 * (median_values[f] - imp_prime[f]) <= c_f[f])
    
            # Score constraint for median aggregation
            for j in range(m):
                constraints.append(score_prime[j] == cp.sum(cp.multiply((imp_prime + median_values) / 2, value_matrix[j, :])))
    
        # Objective: Minimize L1 distance between manipulated score and ideal score
        l1_distance = cp.norm1(score_prime - ideal_score)
        objective = cp.Minimize(l1_distance)
    
        # Solve the LP
        problem = cp.Problem(objective, constraints)
        result = problem.solve(solver=cp.SCIP if elicitation == "approval" or elicitation == "plurality" else cp.SCS)
        
        if problem.status in ["infeasible", "unbounded"]:
            print(problem.status)
            return None  # Problem is infeasible or unbounded, return None instead of failing
        
        return l1_distance.value

    @staticmethod
    def control_by_deletion(simulator, votes, value_matrix, ideal_scores, max_deletions, method):
    

    @staticmethod
    def control_by_cloning(simulator, votes, value_matrix, ideal_scores, max_clones, method):
       
