import unittest
from qwen_agent.agents.advanced_agents.adaptive_learning_agent import AdaptiveLearningAgent

class DummyFeedbackCollector:
    def process(self, feedback):
        return f"Interpreted: {feedback}"

class DummyFineTuner:
    def tune(self, data):
        pass

class TestAdaptiveLearningAgent(unittest.TestCase):
    def setUp(self):
        self.feedback_collector = DummyFeedbackCollector()
        self.fine_tuner = DummyFineTuner()
        self.agent = AdaptiveLearningAgent(feedback_collector=self.feedback_collector, fine_tuner=self.fine_tuner)

    def test_receive_feedback_triggers_adaptation(self):
        self.agent.receive_feedback("Good job")
        # No direct output, but should run without errors

    def test_decide_next_action(self):
        self.agent.perceive("Start")
        action = self.agent.decide_next_action()
        self.assertIsInstance(action, dict)

if __name__ == "__main__":
    unittest.main()
