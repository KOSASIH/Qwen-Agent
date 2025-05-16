import unittest
from qwen_agent.agents.advanced_agents.explainable_ai_agent import ExplainableAIAgent

def dummy_explanation_generator(action):
    return f"Explanation for action: {action.get('type')}"

class TestExplainableAIAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ExplainableAIAgent(explanation_generator=dummy_explanation_generator)

    def test_decide_next_action_includes_explanation(self):
        self.agent.perceive("Explain this")
        result = self.agent.decide_next_action()
        self.assertIn("explanation", result)
        self.assertIn("action", result)

    def test_get_explanations(self):
        self.agent.perceive("Test")
        self.agent.decide_next_action()
        explanations = self.agent.get_explanations()
        self.assertGreater(len(explanations), 0)

if __name__ == "__main__":
    unittest.main()
