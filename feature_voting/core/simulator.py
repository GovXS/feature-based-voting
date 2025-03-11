import numpy as np
from .enums import ElicitationMethod, AggregationMethod
from .utils import generate_clone_combinations, expand_votes, expand_value_matrix

class VotingSimulator:
    def __init__(self,
                 num_voters: int,
                 metrics: List[str],
                 elicitation_method: ElicitationMethod,
                 alpha: float = 1.0):
        self.num_voters = num_voters
        self.metrics = metrics
        self.elicitation_method = elicitation_method
        self.alpha = alpha

    # ... (rest of the simulator methods) ... 