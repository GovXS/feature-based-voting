import unittest
import numpy as np
from feature_voting import VotingSimulator, ElicitationMethod

class TestVotingSimulator(unittest.TestCase):
    def setUp(self):
        self.metrics = ["metric1", "metric2"]
        self.num_voters = 100
        self.simulator = VotingSimulator(
            num_voters=self.num_voters,
            metrics=self.metrics,
            elicitation_method=ElicitationMethod.FRACTIONAL
        )

    def test_vote_generation(self):
        votes = self.simulator.generate_votes()
        self.assertEqual(votes.shape, (self.num_voters, len(self.metrics))) 