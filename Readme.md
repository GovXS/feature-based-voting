# Feature-Based Voting Simulator

This project simulates a feature-based voting system where voters allocate funds across different metrics, and projects receive funding based on their performance in these metrics.

## Features
- Configurable number of voters, projects, and total funds
- Multiple aggregation methods for vote calculation:
  - Arithmetic Mean
  - Median
  - Geometric Mean
  - Quadratic Voting
- Automatic generation of project metrics
- CSV exports of simulation results
- Timestamped runs for easy comparison

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
python simulator.py
```

### Configuration

You can modify the following parameters in `simulator.py`:

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
total_funds = 10_000_000

# Metric ranges
metric_ranges = {
    "daily_users": (500, 2000),
    "transaction_volume": (100000, 1000000),
    "unique_wallets": (400, 1500),
    "tvl": (1000000, 5000000),
    "developer_activity": (10, 100)
}
```

### Output

The simulator creates a new directory for each run under `data/simulation_runs/` with the format:
```
projects_{num_projects}_voters_{num_voters}_funds_{total_funds}_{timestamp}/
├── projects_metrics/
│   └── projects_metrics.csv
└── fund_allocation/
    ├── fund_allocation_arithmetic_mean.csv
    ├── fund_allocation_median.csv
    ├── fund_allocation_geometric_mean.csv
    └── fund_allocation_quadratic.csv
```

### Understanding Results

1. `projects_metrics.csv`: Contains the randomly generated metrics for each project
2. `fund_allocation_{method}.csv`: Shows how funds are allocated to projects using different aggregation methods




