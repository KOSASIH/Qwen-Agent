import unittest
from qwen_agent.agents.advanced_agents.ethical_ai_enforcer_agent import EthicalAIEnforcerAgent

class DummyEthicalGuidelines:
    def check(self, text):
        return "bad" not in text

class DummyBiasDetector:
    def __call__(self, text):
        return "bias" in text

class TestEthicalAIEnforcerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = EthicalAIEnforcerAgent(ethical_guidelines=DummyEthicalGuidelines(), bias_detector=DummyBiasDetector())

    def test_validate_output(self):
        self.assertTrue(self.agent.validate_output("This is good"))
        self.assertFalse(self.agent.validate_output("This is bad"))
        self.assertFalse(self.agent.validate_output("This contains bias"))

    def test_generate_plan_filters_response(self):
        input_text = "Generate bad response"
        plan = self.agent.generate_plan(input_text)
        self.assertIsInstance(plan, list)
        self.assertTrue(any("respond" in step.get("type", "") for step in plan))

if __name__ == "__main__":
    unittest.main()
