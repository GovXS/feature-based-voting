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
    imp_agg_original = compute_aggregated_importance(votes, aggregation)
    scores_original = compute_scores(imp_agg_original, value_matrix)
    l1_original = np.sum(np.abs(scores - scores_original))
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

    return abs(l1_original-l1_distance.value)


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
    imp_agg_original = compute_aggregated_importance(votes, aggregation)
    scores_original = compute_scores(imp_agg_original, value_matrix)
    l1_original = np.sum(np.abs(ideal_score - scores_original))
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

    return abs(l1_original-l1_distance.value)

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


def simulated_annealing_general(
    initial_state, compute_l1, budget, operation, elicitation, imp, val, 
    ranking_matrix=None, max_iter=500, initial_temperature=100, cooling_rate=0.99, epsilon=1e-6
):
    """Simulated annealing for feature deletion and cloning, considering elicitation methods."""
    
    best_solution = initial_state.copy()
    best_l1 = compute_l1(best_solution)
    temp = initial_temperature
    iteration = 0
    no_improvement = 0

    # Store initial row sum (assuming uniform sum across all rows for cumulative voting)
    row_sum_before = np.sum(imp[0])

    while temp > 1 and iteration < max_iter and no_improvement < 100:
        new_solution = best_solution.copy()

        # Determine the number of changes based on budget
        if operation == "deletion":
            num_changes = random.randint(1, min(budget, max(1, len(best_solution))))
            elements_to_modify = random.sample(list(set(range(imp.shape[1])) if not best_solution else best_solution), num_changes)
        else:  # Cloning case
            num_changes = random.randint(1, min(budget, max(1, len(best_solution.keys()))))
            elements_to_modify = random.sample(list(best_solution.keys()), num_changes)

        # Apply changes for deletion
        if operation == "deletion":
            for element in elements_to_modify:
                if element in new_solution:
                    new_solution.remove(element)
                else:
                    new_solution.add(element)

            # Apply proper vote transfer for Plurality voting
            if elicitation == "plurality":
                imp_new = transfer_votes_after_deletion(imp, ranking_matrix, new_solution)

            # Normalize for Cumulative voting
            elif elicitation == "cumulative":
                remaining_features = sorted(set(range(imp.shape[1])) - new_solution)
                imp_new = imp[:, remaining_features]
                imp_new = imp_new / np.sum(imp_new, axis=1, keepdims=True)
                imp_new *= row_sum_before  # Restore original sum before normalization

        # Apply changes for cloning
        elif operation == "cloning":
            for element in elements_to_modify:
                if random.random() < 0.5 and new_solution[element] > 0:
                    new_solution[element] -= 1
                elif sum(new_solution.values()) < budget:
                    new_solution[element] += 1

            imp_new = np.copy(imp)
            
            if elicitation == "fractional":
                pass  # Clones receive the same weight as the original metric
            
            elif elicitation == "cumulative":
                for feature, clone_count in new_solution.items():
                    if clone_count > 0:
                        total_parts = 1 + clone_count
                        imp_new[:, feature] /= total_parts
                        cloned_values = np.tile(imp[:, feature] / total_parts, (clone_count, 1)).T
                        imp_new = np.column_stack((imp_new, cloned_values))
                imp_new = imp_new / np.sum(imp_new, axis=1, keepdims=True)
                imp_new *= row_sum_before  # Restore original sum before normalization

            elif elicitation == "approval":
                for feature, clone_count in new_solution.items():
                    for _ in range(clone_count):
                        imp_new = np.column_stack((imp_new, (imp[:, feature] == 1).astype(int)))

            elif elicitation == "plurality":
                for feature, clone_count in new_solution.items():
                    for i in range(imp.shape[0]):
                        if imp[i, feature] == 1:
                            move_to_clone = np.random.randint(0, clone_count + 1)
                            if move_to_clone > 0:
                                imp_new[i, feature] = 0
                                imp_new[i, -clone_count + move_to_clone - 1] = 1  # Assign to a clone

        # Compute new L1 distance
        new_l1 = compute_l1(new_solution)
        if new_l1 < best_l1:
            best_solution = new_solution
            best_l1 = new_l1
            no_improvement = 0
        else:
            no_improvement += 1
        
        if abs(best_l1 - new_l1) < epsilon:
            break
        
        temp *= cooling_rate
        iteration += 1

    return best_solution, best_l1


def control_by_deletion(imp, val, ideal_scores, budget, elicitation, aggregation):
    """Simulated annealing for control by deletion with budget constraint."""
    imp_agg_original = compute_aggregated_importance(imp, aggregation)
    scores_original = compute_scores(imp_agg_original, val)
    l1_original = l1_distance(scores_original, ideal_scores)
    remaining_features = set(range(imp.shape[1]))
    ranking_matrix = generate_ranking_matrix(imp) if elicitation == "plurality" else None

    deleted_features, best_l1 = simulated_annealing_general(
        set(),
        lambda df: compute_l1_deletion(df, imp, val, ranking_matrix, elicitation, aggregation, ideal_scores),
        budget,
        "deletion",
        elicitation,
        imp,
        val,
        ranking_matrix,
        max_iter=500
    )
    return abs(l1_original-best_l1)

def control_by_cloning(imp, val, ideal_scores, budget, elicitation, aggregation):
    """Simulated annealing for control by cloning with budget constraint."""
    imp_agg_original = compute_aggregated_importance(imp, aggregation)
    scores_original = compute_scores(imp_agg_original, val)
    l1_original = l1_distance(scores_original, ideal_scores)
    initial_cloning_plan = {f: 0 for f in range(imp.shape[1])}

    cloned_plan, best_l1 = simulated_annealing_general(
        initial_cloning_plan,
        lambda cp: compute_l1_cloning(cp, imp, val, elicitation, aggregation, ideal_scores),
        budget,
        "cloning",
        elicitation,
        imp,
        val,
        max_iter=500
    )
    return abs(l1_original-best_l1)

def compute_l1_cloning(cloning_plan, imp, val, elicitation, aggregation, ideal_scores):
    """Compute L1 distance after cloning operation."""
    imp_new = np.copy(imp)
    val_new = np.copy(val)
    
    if elicitation == "fractional":
        pass  # Clones receive the same weight as the original metric
    
    elif elicitation == "cumulative":
        for feature, clone_count in cloning_plan.items():
            if clone_count > 0:
                total_parts = 1 + clone_count
                imp_new[:, feature] /= total_parts
                cloned_values = np.tile(imp[:, feature] / total_parts, (clone_count, 1)).T
                imp_new = np.column_stack((imp_new, cloned_values))
                val_new = np.column_stack((val_new, np.tile(val[:, feature], (clone_count, 1)).T))
        imp_new = imp_new / np.sum(imp_new, axis=1, keepdims=True)
    
    elif elicitation == "approval":
        new_columns = []
        new_val_columns = []
        for feature, clone_count in cloning_plan.items():
            for _ in range(clone_count):
                new_columns.append((imp[:, feature] == 1).astype(int))
                new_val_columns.append(val[:, feature])
        
        if new_columns:
            imp_new = np.column_stack((imp_new, np.column_stack(new_columns)))
            val_new = np.column_stack((val_new, np.column_stack(new_val_columns)))
    
    elif elicitation == "plurality":
        new_columns = []
        new_val_columns = []
        for feature, clone_count in cloning_plan.items():
            for i in range(imp.shape[0]):
                if imp[i, feature] == 1:
                    move_to_clone = np.random.randint(0, clone_count + 1)
                    if move_to_clone > 0:
                        imp_new[i, feature] = 0
                        imp_new[i, -clone_count + move_to_clone - 1] = 1  # Assign to a clone
                        new_columns.append((imp[:, feature] == 1).astype(int))
                        new_val_columns.append(val[:, feature])
        
        if new_columns:
            imp_new = np.column_stack((imp_new, np.column_stack(new_columns)))
            val_new = np.column_stack((val_new, np.column_stack(new_val_columns)))
    
    imp_agg_new = compute_aggregated_importance(imp_new, aggregation)
    new_scores = compute_scores(imp_agg_new, val_new)
    
    return l1_distance(new_scores, ideal_scores)

def compute_l1_deletion(deleted_features, imp, val, ranking_matrix, elicitation, aggregation, ideal_scores):
    """Compute L1 distance after deletion operation."""
    remaining_features = sorted(set(range(imp.shape[1])) - deleted_features)
    imp_new = imp[:, remaining_features]
    val_new = val[:, remaining_features]
    
    if elicitation == "plurality":
        imp_new = transfer_votes_after_deletion(imp, ranking_matrix, deleted_features)
    
    if elicitation == "cumulative":
        imp_new = imp_new / np.sum(imp_new, axis=1, keepdims=True)
    
    imp_agg_new = compute_aggregated_importance(imp_new, aggregation)
    new_scores = compute_scores(imp_agg_new, val_new)
    
    return l1_distance(new_scores, ideal_scores)
