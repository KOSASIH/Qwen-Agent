import unittest
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class DummyMemoryManager:
    def __init__(self):
        self.storage = {}

    def add_memory(self, observation, modalities=None):
        self.storage['last'] = observation

    def store(self, key, value):
        self.storage[key] = value

    def retrieve(self, key):
        return self.storage.get(key, None)

class DummyTool:
    def __call__(self, **kwargs):
        return {"result": "tool_executed", "params": kwargs}

class TestBaseAdvancedAgent(unittest.TestCase):
    def setUp(self):
        self.memory_manager = DummyMemoryManager()
        self.tool_manager = {"dummy_tool": DummyTool()}
        self.agent = BaseAdvancedAgent(memory_manager=self.memory_manager, tool_manager=self.tool_manager)

    def test_perceive_and_remember(self):
        self.agent.perceive("Test observation")
        self.agent.remember("key1", "value1")
        recalled = self.agent.recall("key1")
        self.assertEqual(recalled, "value1")
        self.assertEqual(self.memory_manager.storage['last'], "Test observation")

    def test_invoke_tool(self):
        result = self.agent.invoke_tool("dummy_tool", {"param1": 123})
        self.assertIn("result", result)
        self.assertEqual(result["params"]["param1"], 123)

    def test_decide_next_action(self):
        self.agent.perceive("Hello")
        action = self.agent.decide_next_action()
        self.assertIsInstance(action, dict)

if __name__ == "__main__":
    unittest.main()
