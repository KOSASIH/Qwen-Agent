import unittest
from qwen_agent.agents.advanced_agents.multi_agent_coordinator import MultiAgentCoordinator
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class DummyAgent(BaseAdvancedAgent):
    def __init__(self):
        super().__init__()
    def decide_next_action(self):
        return {"type": "response", "content": "Handled by DummyAgent"}

class TestMultiAgentCoordinator(unittest.TestCase):
    def setUp(self):
        self.dummy_agent = DummyAgent()
        self.coordinator = MultiAgentCoordinator(agents={"analyze": self.dummy_agent})

    def test_delegate_task(self):
        task = {"type": "analyze", "content": "Analyze this"}
        result = self.coordinator.delegate_task(task)
        self.assertEqual(result["content"], "Handled by DummyAgent")

    def test_generate_plan(self):
        plan = self.coordinator.generate_plan("Analyze this. Another task.")
        self.assertIsInstance(plan, list)
        self.assertTrue(any("delegated_result" == item.get("type") for item in plan))

if __name__ == "__main__":
    unittest.main()
