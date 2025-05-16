import unittest
from qwen_agent.agents.advanced_agents.memorized_agent import MemorizedAgent

class DummyMemoryManager:
    def __init__(self):
        self.data = {}
    def store(self, key, val):
        self.data[key] = val
    def retrieve(self, key):
        return self.data.get(key, None)
    def get_recent(self, num_entries=10):
        return list(self.data.values())[-num_entries:]

class TestMemorizedAgent(unittest.TestCase):
    def setUp(self):
        self.memory_manager = DummyMemoryManager()
        self.agent = MemorizedAgent(memory_manager=self.memory_manager)

    def test_add_and_retrieve_memory(self):
        self.agent.add_to_memory("test_key", "test_val")
        val = self.agent.retrieve_from_memory("test_key")
        self.assertEqual(val, "test_val")

    def test_recall_contextual_memory(self):
        self.agent.add_to_memory("k1", "mem1")
        self.agent.add_to_memory("k2", "mem2")
        context = self.agent.recall_contextual_memory()
        self.assertIn("mem1", context)
        self.assertIn("mem2", context)

    def test_generate_plan_includes_memory(self):
        plan = self.agent.generate_plan("User query")
        self.assertIsInstance(plan, list)
        self.assertTrue(any("respond" in act.get("type","") for act in plan))

if __name__ == "__main__":
    unittest.main()
