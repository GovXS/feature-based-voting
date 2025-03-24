# Feature-Based Voting Simulator

This project simulates a feature-based voting system where voters allocate funds across different metrics, and projects receive funding based on their performance in these metrics.


## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd feature-based-voting-simulator
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulator

Run the simulator with default parameters:
```bash
python main.py
```

### Configuration

You can modify the following parameters in `main.py`:

```python
# Key simulation parameters
metrics = [
    "daily_users",
    "transaction_volume",
    "unique_wallets",
    "tvl",
    "developer_activity"
]
num_projects = 100
num_voters = 300


# Metric ranges
metric_ranges = 10000
```

# Experiments

## Aggregation Combinations

1. Bribery Resistance:
Approval elicitation methods show the highest resistance to bribery, with both arithmetic mean and median aggregations.
Fractional elicitation also shows considerable resistance, though lower than approval.
Cumulative elicitation shows the lowest resistance, indicating susceptibility to bribery.


2. Manipulation Resistance:
Again, approval elicitation stands out with the highest manipulation resistance.
Fractional elicitation has moderate-to-high resistance.
Cumulative elicitation methods again show significantly lower resistance, meaning they're vulnerable to manipulation.
3. Deletion Resistance:
The approval and fractional methods have higher resistance compared to cumulative and plurality.
Cumulative elicitation shows negligible resistance, essentially making it very susceptible to feature deletion strategies.
4. Cloning Resistance:
All methods show virtually zero resistance to feature cloning, indicating that cloning features strongly affect the outcomes, regardless of the elicitation or aggregation method.

### Key Takeaways:
Approval elicitation consistently provides the highest resilience to bribery, manipulation, and deletion strategies, making it the strongest in resisting strategic voting behavior.
Fractional elicitation also performs well overall but is slightly less resistant than approval.
Cumulative elicitation is the most vulnerable, providing the least resistance to strategic manipulations.

### Ranked Method Combinations (Overall Resistance):
The combined resistance scores (sum of all resistance measures) clearly rank method combinations as follows:

Approval (Arithmetic Mean): Highest combined resistance score (211.77).
Approval (Median): Second highest combined resistance (207.20).
Fractional (Median): High combined resistance (178.03).
Fractional (Arithmetic Mean): High combined resistance (176.57).
Cumulative (Median) and Cumulative (Arithmetic Mean): Significantly lower resistance.
Plurality (Arithmetic Mean & Median): Lowest combined resistance, particularly weak against manipulation and deletion.

### Recommendations based on the analysis:
Approval elicitation with arithmetic mean aggregation emerges as the most robust and consistently resistant option against strategic manipulations.
Fractional elicitation with median or mean aggregation could also be suitable if a continuous approach is preferred.


## Alpha Senstivity
Regarding the experiment Bigger alpha-> more heterogenous agents.
I think it could be interesting to test a few alphas:
for each one generate lets say 100 instances given an elicitation and for each instance run it on all combinations of aggregation and  problem , take the mean of it (for every instance) and then take the mean of all 100 instances. Then we could see how alpha effects strategic behaviors (not specific one) for a combination of elicitation and aggregation 

1. Bribery Resistance by Alpha:
Geometric mean aggregation shows significantly higher resistance to bribery across elicitation methods compared to arithmetic mean.
Resistance tends to increase slightly or remain relatively stable as alpha increases, indicating more heterogeneous voters may modestly enhance bribery resistance, particularly under geometric mean aggregation.
2. Manipulation Resistance by Alpha:
Resistance values are similar across elicitation methods and aggregation techniques, with a slight increasing trend as alpha grows.
Higher alpha values (greater voter heterogeneity) appear to strengthen resistance to manipulation slightly, suggesting that heterogeneity might protect against voter manipulation.
3. Deletion Resistance by Alpha:
Deletion resistance is low across all methods, though minor fluctuations occur.
There's no clear increasing or decreasing trend, suggesting alpha does not significantly impact resistance to feature deletion strategies.
4. Cloning Resistance by Alpha:
Cloning resistance remains consistently at zero across all alpha values and method combinations, reinforcing earlier findings that cloning strategies consistently succeed regardless of voter heterogeneity.
5. Aggregate Resistance by Alpha:
Aggregate resistance (mean of bribery, manipulation, and deletion resistance) generally increases with higher alpha values.
Approval and fractional elicitation consistently exhibit superior aggregate resistance compared to cumulative.

## Key Takeaways:
Higher alpha values (more heterogeneous voters) slightly enhance resistance to bribery and manipulation, particularly with geometric mean aggregation.
Geometric mean aggregation clearly outperforms arithmetic mean in resisting bribery.
Cloning resistance is unaffected by voter heterogeneity, consistently showing very low resistance.
Approval and fractional elicitation methods are consistently more robust compared to cumulative elicitation.
Deletion resistance remains low across all methods, and cloning resistance remains consistently negligible



## Data Manipulation

Instead of voters manipulating, projects modify their feature values to get better rankings.
Why is it Important?
In retroactive funding, projects may inflate metrics to get higher funding.
It tests whether project rankings remain fair under strategic feature adjustments.
How to Implement?
Select a few projects.
Modify their feature values within a reasonable range (e.g., increase by 10%).
Compare score changes before and after modification.

Data manipulation in metrics-based voting can occur when candidates artificially inflate their scores or selectively optimize certain metrics to game the system. This is particularly problematic in scenarios where candidates learn how the evaluation algorithm works and adjust their behavior to maximize their ranking without genuinely improving quality or performance. 

For example, an open-source funding system might measure “commits to a code repository”, and add a time decay for developer rewards so that inactive developers receive less than those who contributed more recently. With this mechanism in mind, developers may submit frequent but low-value commits to maintain eligibility. O teams may delay merging contributions to ensure that rewards fall within an optimal timeframe.

1. Average Score Change:
Approval and fractional elicitation methods show the highest sensitivity, meaning they are more susceptible to significant score changes due to strategic feature manipulation.
Cumulative elicitation shows moderate sensitivity, with lower average score changes, indicating slightly higher robustness against manipulation.
2. Average Rank Change:
Fractional elicitation methods experience the greatest number of rank changes after data manipulation, highlighting significant sensitivity to strategic adjustments.
Cumulative and approval elicitation methods have slightly fewer average rank changes compared to fractional, but still exhibit noticeable susceptibility.

### Key Insights:
Fractional elicitation methods are particularly vulnerable to strategic project feature manipulation, affecting both project scores and rankings considerably.
Approval elicitation also shows notable susceptibility.
Cumulative elicitation methods appear more resistant compared to the others, indicating greater stability against strategic project modifications.

1. Distribution of Score Changes:
Fractional elicitation methods exhibit broader distributions, indicating substantial variability in how project scores respond to manipulation.
Approval elicitation methods also show considerable variation but slightly less than fractional methods.
Cumulative elicitation methods consistently demonstrate narrower distributions, highlighting their relative robustness against manipulation in terms of score changes.
2. Distribution of Rank Changes:
Fractional elicitation shows the most considerable variability and highest overall sensitivity regarding rank changes, suggesting a high susceptibility to strategic manipulation.
Approval elicitation methods have moderate variability, suggesting somewhat stable rankings, yet still vulnerable.
Cumulative elicitation methods have the lowest variability, emphasizing greater stability in maintaining project rankings despite strategic manipulations.
3. Relationship between Score Change and Rank Change:
A positive correlation between score changes and rank changes is clearly visible, meaning projects that experience significant score adjustments due to feature manipulation also tend to experience notable changes in rankings.
This correlation is strongest within fractional elicitation, underlining its vulnerability to strategic project modifications.
Approval and cumulative methods show somewhat less pronounced relationships, further confirming their relative stability.

### Recommendations based on analysis:
For systems susceptible to strategic project feature manipulation, cumulative elicitation methods demonstrate the greatest stability and robustness.
Conversely, fractional elicitation methods should be used cautiously or with additional safeguards against manipulation



