import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple
import time
import itertools
import random

  
def bribery_optimization(votes, value_matrix, scores, budget, elicitation, aggregation):

    n, k = votes.shape  # Number of voters and features
    m = value_matrix.shape[0]  # Number of projects

    imp_prime = cp.Variable((n, k))

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


def manipulation(votes, value_matrix, elicitation,aggregation):
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
    imp_prime = cp.Variable((n, k))
    m = value_matrix.shape[0]  # Number of projects
    random_agent_index = np.random.choice(n)
    score_prime = cp.Variable(m)  # New manipulated scores
    # Compute the ideal score for the manipulator
    ideal_score = value_matrix @ votes[random_agent_index, :]

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
        imp_agg = (cp.sum(votes, axis=0) - votes[random_agent_index, :] + imp_prime) / n

        # Compute the new project scores
        for j in range(m):
            constraints.append(score_prime[j] == cp.sum(cp.multiply(imp_agg, value_matrix[j, :])))

    elif aggregation == "median":
        # Compute the median values excluding the manipulator
        votes_excluding_agent = np.delete(votes, random_agent_index, axis=0)
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

def generate_ranking_matrix(imp):
    """Generate ranking preferences for Plurality voting."""
    n, k = imp.shape
    ranking = np.zeros((n, k), dtype=int)

    for i in range(n):
        voted_feature = np.argmax(imp[i])
        available_ranks = list(range(2, k + 1))
        np.random.shuffle(available_ranks)

        for f in range(k):
            if f == voted_feature:
                ranking[i, f] = 1
            else:
                ranking[i, f] = available_ranks.pop()

    return ranking

def compute_aggregated_importance(imp, aggregation):
    """Compute aggregated importance vector using mean or median."""
    return np.mean(imp, axis=0) if aggregation == "mean" else np.median(imp, axis=0)

def compute_scores(imp_agg, val):
    """Compute project scores using the importance vector and value matrix."""
    return np.dot(val, imp_agg)

def l1_distance(scores, ideal_scores):
    """Compute L1 distance between actual and ideal scores."""
    return np.sum(np.abs(scores - ideal_scores))

def transfer_votes_after_deletion(imp, ranking_matrix, deleted_features):
    """For Plurality voting, reassign votes to the next ranked feature."""
    n, k = imp.shape
    new_imp = np.zeros_like(imp)

    for i in range(n):
        original_vote = np.argmax(imp[i])
        if original_vote in deleted_features:
            for rank in range(2, k + 1):
                next_feature = np.where(ranking_matrix[i] == rank)[0][0]
                if next_feature not in deleted_features:
                    new_imp[i, next_feature] = 1
                    break
        else:
            new_imp[i, original_vote] = 1

    return new_imp[:, sorted(set(range(k)) - deleted_features)]

def simulated_annealing(feature_set, compute_l1, max_iter=100, initial_temperature=100, cooling_rate=0.99):
    """Simulated annealing to find the best feature subset minimizing L1 distance."""
    feature_list = sorted(list(feature_set))  # Convert set to list
    best_solution = set(random.sample(feature_list, min(len(feature_list), max_iter // 10)))
    best_l1 = compute_l1(best_solution)

    temp = initial_temperature
    iteration = 0

    while temp > 1 and iteration < max_iter:
        new_solution = best_solution.copy()

        if len(new_solution) > 1 and random.random() < 0.5:
            new_solution.remove(random.choice(list(new_solution)))
        else:
            new_solution.add(random.choice(feature_list))

        new_l1 = compute_l1(new_solution)

        if new_l1 < best_l1 or random.random() < np.exp((best_l1 - new_l1) / temp):
            best_solution = new_solution
            best_l1 = new_l1

        temp *= cooling_rate
        iteration += 1

    return best_solution, best_l1



def control_by_deletion(imp, val, ideal_scores, elicitation, aggregation):
    """Simulated annealing for control by deletion."""
    start_time = time.time()
    k = imp.shape[1]
    best_l1 = float('inf')
    remaining_features = set(range(k))

    if elicitation == "plurality":
        ranking_matrix = generate_ranking_matrix(imp)

    def compute_l1(deleted_features):
        """Compute L1 distance given a set of deleted features."""
        remaining = sorted(remaining_features - deleted_features)
        imp_new = imp[:, remaining]
        val_new = val[:, remaining]

        if elicitation == "cumulative":
            imp_new /= np.sum(imp_new, axis=1, keepdims=True)

        if elicitation == "plurality":
            imp_new = transfer_votes_after_deletion(imp, ranking_matrix, deleted_features)

        imp_agg_new = compute_aggregated_importance(imp_new, aggregation)
        new_scores = compute_scores(imp_agg_new, val_new)
        return l1_distance(new_scores, ideal_scores)

    deleted_features, best_l1 = simulated_annealing(remaining_features, compute_l1, max_iter=200)

    return best_l1

def control_by_cloning(imp, val, ideal_scores,elicitation, aggregation):
    """Simulated annealing for control by cloning."""
    start_time = time.time()
    n, k = imp.shape
    best_l1 = float('inf')
    feature_set = set(range(k))

    if elicitation == "plurality":
        ranking_matrix = generate_ranking_matrix(imp)

    def compute_l1(cloned_features):
        """Compute L1 distance given a set of cloned features."""
        imp_new = np.copy(imp)
        val_new = np.copy(val)

       # Convert set to list of integers (Change 1)
        cloned_features = list(cloned_features)
    
        num_clones = len(cloned_features)

        for f in cloned_features:
            imp_new = np.column_stack((imp_new, imp[:, f]))  # Clone feature
            val_new = np.column_stack((val_new, val[:, f]))

        if elicitation == "cumulative":
            imp_new[:, cloned_features] /= (1 + num_clones)  # Normalize original feature
            #Change 2
            #imp_new[:, -num_clones:] = imp[:, list(cloned_features)][:, np.newaxis] / (1 + num_clones)  # Normalize clones
            imp_new[:, -num_clones:] = imp[:, cloned_features] / (1 + num_clones)  # Remove np.newaxis

        elif elicitation == "plurality":
            for i in range(n):
                if imp[i, f] == 1:
                    move_to_clone = np.random.choice([True, False])
                    if move_to_clone:
                        imp_new[i, f] = 0
                        imp_new[i, -num_clones:] = 1

        imp_agg_new = compute_aggregated_importance(imp_new, aggregation)
        new_scores = compute_scores(imp_agg_new, val_new)
        return l1_distance(new_scores, ideal_scores)

    cloned_features, best_l1 = simulated_annealing(feature_set, compute_l1, max_iter=200)

    return best_l1