import unittest
from qwen_agent.agents.advanced_agents.contextual_reasoning_agent import ContextualReasoningAgent

def mock_reasoning_engine(text):
    return [{"type": "respond", "content": f"Reasoned response to: {text}"}]

class TestContextualReasoningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ContextualReasoningAgent(reasoning_engine=mock_reasoning_engine)

    def test_generate_plan_uses_reasoning(self):
        plan = self.agent.generate_plan("Analyze problem")
        self.assertIsInstance(plan, list)
        self.assertTrue(any("respond" in act.get("type", "") for act in plan))

    def test_infer_returns_confidence(self):
        result = self.agent.infer(["Fact 1 about AI", "Fact 2"], "AI")
        self.assertIn("confidence", result)
        self.assertIn("explanation", result)

if __name__ == "__main__":
    unittest.main()
