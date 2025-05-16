import unittest
from qwen_agent.agents.advanced_agents.self_improving_agent import SelfImprovingAgent

class DummyPerformanceTracker:
    def __init__(self):
        self.actions = []
    def record_action(self, action):
        self.actions.append(action)
    def evaluate(self, history):
        return {"score": len(history)}

class DummyImprovementStrategy:
    def tune(self, eval_data):
        pass

class TestSelfImprovingAgent(unittest.TestCase):
    def setUp(self):
        self.tracker = DummyPerformanceTracker()
        self.strategy = DummyImprovementStrategy()
        self.agent = SelfImprovingAgent(performance_tracker=self.tracker, improvement_strategy=self.strategy)

    def test_decide_next_action_records(self):
        self.agent.perceive("Input")
        action = self.agent.decide_next_action()
        self.assertIn(action, self.agent.history)

    def test_evaluate_performance_applies_improvements(self):
        self.agent.history = [{"type": "test_action"}]
        self.agent.evaluate_performance()

if __name__ == "__main__":
    unittest.main()
