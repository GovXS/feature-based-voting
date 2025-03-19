import pandas as pd
import os

def save_simulation_results(results_dir, sim_params, votes, value_matrix, ideal_scores, results):
    # Convert sim_params to a format that pd.DataFrame.from_dict can handle
    sim_params_cleaned = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in sim_params.items()}

    # Save simulation parameters
    pd.DataFrame.from_dict(sim_params_cleaned, orient="index", columns=["Value"]).to_csv(os.path.join(results_dir, "sim_params.csv"))

    # Save votes
    pd.DataFrame(votes).to_csv(os.path.join(results_dir, "votes.csv"), index=False)

    # Save value matrix
    pd.DataFrame(value_matrix).to_csv(os.path.join(results_dir, "value_matrix.csv"), index=False)

    # Save ideal scores
    pd.DataFrame(ideal_scores, columns=["ideal_scores"]).to_csv(os.path.join(results_dir, "ideal_scores.csv"), index=False)

    # Save results
    #pd.DataFrame.from_dict(results, orient="index", columns=["Minimum L1 Distance"]).to_csv(os.path.join(results_dir, "results.csv"))


