import unittest
import numpy as np
from feature_voting import VotingSimulator, VotingOptimizer, ElicitationMethod

class TestVotingOptimizer(unittest.TestCase):
    def setUp(self):
        self.metrics = ["metric1", "metric2"]
        self.num_voters = 100
        self.simulator = VotingSimulator(
            num_voters=self.num_voters,
            metrics=self.metrics,
            elicitation_method=ElicitationMethod.FRACTIONAL
        )
        self.votes = self.simulator.generate_votes()
        self.value_matrix = np.random.uniform(0, 1, size=(5, len(self.metrics)))
        self.ideal_scores = np.random.uniform(0, 1, size=5)

    def test_bribery_optimization(self):
        budget = 10.0
        result = VotingOptimizer.bribery_optimization(
            self.simulator, self.votes, self.value_matrix, 
            self.ideal_scores, budget)
        self.assertIsInstance(result, tuple) 