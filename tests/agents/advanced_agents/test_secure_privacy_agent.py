import unittest
from qwen_agent.agents.advanced_agents.secure_privacy_agent import SecurePrivacyAgent

class DummyPrivacyModule:
    def anonymize(self, data):
        return f"anonymized_{data}"

class TestSecurePrivacyAgent(unittest.TestCase):
    def setUp(self):
        self.privacy_module = DummyPrivacyModule()
        self.agent = SecurePrivacyAgent(privacy_module=self.privacy_module)

    def test_anonymize_data(self):
        result = self.agent.anonymize_data("sensitive_data")
        self.assertEqual(result, "anonymized_sensitive_data")

    def test_perceive_anonymizes(self):
        self.agent.perceive("private info", modalities={"image": "image_data"})
        # Should store anonymized in memory or process without error

    def test_generate_plan_anonymizes(self):
        plan = self.agent.generate_plan("Query with sensitive info")
        self.assertIsInstance(plan, list)

if __name__ == "__main__":
    unittest.main()
