import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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


def visualize_combinations_experiment_results(results_dir, summary_results_df,detailed_results_df):
    
    # Set up plot aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # Plot Bribery Resistance
    plt.subplot(2, 2, 1)
    sns.barplot(data=summary_results_df, x='elicitation', y='bribery_resistance', hue='aggregation')
    plt.title('Average Bribery Resistance')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average L1 Distance')

    # Plot Manipulation Resistance
    plt.subplot(2, 2, 2)
    sns.barplot(data=summary_results_df, x='elicitation', y='manipulation_resistance', hue='aggregation')
    plt.title('Average Manipulation Resistance')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average L1 Distance')

    # Plot Deletion Resistance
    plt.subplot(2, 2, 3)
    sns.barplot(data=summary_results_df, x='elicitation', y='deletion_resistance', hue='aggregation')
    plt.title('Average Deletion Resistance')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average L1 Distance')

    # Plot Cloning Resistance
    plt.subplot(2, 2, 4)
    sns.barplot(data=summary_results_df, x='elicitation', y='cloning_resistance', hue='aggregation')
    plt.title('Average Cloning Resistance')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average L1 Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "average_resistance_scores.png"))  # Save Plot 1
    plt.show()

    # Plot detailed distributions to understand variability better
    plt.figure(figsize=(14, 12))

    # Bribery resistance distribution
    plt.subplot(2, 2, 1)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='bribery_resistance', hue='aggregation')
    plt.title('Bribery Resistance Distribution')
    plt.xlabel('Elicitation Method')
    plt.ylabel('L1 Distance')

    # Manipulation resistance distribution
    plt.subplot(2, 2, 2)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='manipulation_resistance', hue='aggregation')
    plt.title('Manipulation Resistance Distribution')
    plt.xlabel('Elicitation Method')
    plt.ylabel('L1 Distance')

    # Deletion resistance distribution
    plt.subplot(2, 2, 3)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='deletion_resistance', hue='aggregation')
    plt.title('Deletion Resistance Distribution')
    plt.xlabel('Elicitation Method')
    plt.ylabel('L1 Distance')

    # Cloning resistance distribution
    plt.subplot(2, 2, 4)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='cloning_resistance', hue='aggregation')
    plt.title('Cloning Resistance Distribution')
    plt.xlabel('Elicitation Method')
    plt.ylabel('L1 Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "resistance_score_distributions.png"))  # Save Plot 2
    plt.show()

    # Rank methods based on combined resistance scores
    summary_results_df['combined_resistance'] = (summary_results_df['bribery_resistance'] + 
                                                summary_results_df['manipulation_resistance'] + 
                                                summary_results_df['deletion_resistance'] + 
                                                summary_results_df['cloning_resistance'])

    # Sort to identify the best performing combination
    ranked_combinations = summary_results_df.sort_values(by='combined_resistance', ascending=False)

    # Save ranked combinations to CSV
    ranked_combinations.to_csv(os.path.join(results_dir, "ranked_combinations.csv"), index=False)

    # Re-attempt pairplot using histograms instead of KDE to avoid errors with zero variance
    sns.pairplot(detailed_results_df, hue='elicitation', diag_kind='hist',
                vars=['bribery_resistance', 'manipulation_resistance', 
                    'deletion_resistance', 'cloning_resistance'])

    plt.suptitle('Pairplot of Resistance Measures by Elicitation Method', y=1.02)
    plt.savefig(os.path.join(results_dir, "pairplot.png"))  # Save Plot 2
    plt.show()


def visualize_alpha_sensitivity_experiment_results(results_dir,alpha_results_df):

    # Set up the plot aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # Plot Bribery Resistance by alpha
    plt.subplot(2, 2, 1)
    sns.lineplot(data=alpha_results_df, x='alpha', y='bribery_l1', hue='elicitation', style='aggregation', markers=True, dashes=False)
    plt.title('Bribery Resistance by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Average L1 Distance')

    # Plot Manipulation Resistance by alpha
    plt.subplot(2, 2, 2)
    sns.lineplot(data=alpha_results_df, x='alpha', y='manipulation_l1', hue='elicitation', style='aggregation', markers=True, dashes=False)
    plt.title('Manipulation Resistance by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Average L1 Distance')

    # Plot Deletion Resistance by alpha
    plt.subplot(2, 2, 3)
    sns.lineplot(data=alpha_results_df, x='alpha', y='deletion_l1', hue='elicitation', style='aggregation', markers=True, dashes=False)
    plt.title('Deletion Resistance by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Average L1 Distance')

    # Cloning Resistance by alpha (likely flat as seen before)
    plt.subplot(2, 2, 4)
    sns.lineplot(data=alpha_results_df, x='alpha', y='cloning_li', hue='elicitation', style='aggregation', markers=True, dashes=False)
    plt.title('Cloning Resistance by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Average L1 Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_1.png"))  # Save Plot 1
    plt.show()

    # Additional visualizations to better explore the alpha sensitivity experiment results

    # Violin plots to explore distribution across alpha values and method combinations
    plt.figure(figsize=(16, 12))

    # Bribery resistance distributions
    plt.subplot(2, 2, 1)
    sns.violinplot(data=alpha_results_df, x='alpha', y='bribery_l1', hue='elicitation', split=True)
    plt.title('Bribery Resistance Distributions by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Bribery Resistance (L1)')

    # Manipulation resistance distributions
    plt.subplot(2, 2, 2)
    sns.violinplot(data=alpha_results_df, x='alpha', y='manipulation_l1', hue='elicitation', split=True)
    plt.title('Manipulation Resistance Distributions by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Manipulation Resistance (L1)')

    # Deletion resistance distributions
    plt.subplot(2, 2, 3)
    sns.violinplot(data=alpha_results_df, x='alpha', y='deletion_l1', hue='elicitation', split=True)
    plt.title('Deletion Resistance Distributions by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Deletion Resistance (L1)')

    # Aggregate resistance comparison (excluding cloning as it is consistently zero)
    alpha_results_df['aggregate_resistance'] = alpha_results_df[['bribery_l1', 'manipulation_l1', 'deletion_l1']].mean(axis=1)
    plt.subplot(2, 2, 4)
    sns.lineplot(data=alpha_results_df, x='alpha', y='aggregate_resistance', hue='elicitation', style='aggregation', markers=True, dashes=False)
    plt.title('Aggregate Resistance by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Aggregate Resistance (Average L1 Distance)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_2.png"))  # Save Plot 1
    plt.show()

def visualize_data_manipulation_results(results_dir,summary_results_df, detailed_results_df):
    # Import visualization libraries again after environment reset
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up plot aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6))

    # Visualization for Score Changes due to Data Manipulation
    plt.subplot(1, 2, 1)
    sns.barplot(data=summary_results_df, x='elicitation', y='score_change', hue='aggregation')
    plt.title('Average Score Change Due to Data Manipulation')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average Score Change')

    # Visualization for Rank Changes due to Data Manipulation
    plt.subplot(1, 2, 2)
    sns.barplot(data=summary_results_df, x='elicitation', y='rank_change', hue='aggregation')
    plt.title('Average Rank Change Due to Data Manipulation')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Average Rank Change')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_1.png"))  # Save Plot 1
    plt.show()

    # Additional analyses: Distribution visualization and detailed relationships

    plt.figure(figsize=(16, 10))

    # Distribution of Score Changes
    plt.subplot(2, 2, 1)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='score_change', hue='aggregation')
    plt.title('Distribution of Score Changes by Method')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Score Change')

    # Distribution of Rank Changes
    plt.subplot(2, 2, 2)
    sns.boxplot(data=detailed_results_df, x='elicitation', y='rank_change', hue='aggregation')
    plt.title('Distribution of Rank Changes by Method')
    plt.xlabel('Elicitation Method')
    plt.ylabel('Rank Change')

    # Relationship between Score Change and Rank Change
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=detailed_results_df, x='score_change', y='rank_change', hue='elicitation', style='aggregation')
    plt.title('Relationship between Score Change and Rank Change')
    plt.xlabel('Score Change')
    plt.ylabel('Rank Change')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_2.png"))  # Save Plot 1
    plt.show()




