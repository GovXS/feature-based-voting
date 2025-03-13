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




