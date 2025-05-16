import unittest
from qwen_agent.agents.advanced_agents.autonomous_planner_agent import AutonomousPlannerAgent

def mock_plan_generator(input_text):
    return [
        {"type": "tool_use", "tool_name": "search", "parameters": {"query": input_text}},
        {"type": "respond", "content": "Mocked response."}
    ]

class DummyTool:
    def __call__(self, **kwargs):
        return {"result": "search_executed", "params": kwargs}

class TestAutonomousPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AutonomousPlannerAgent(plan_generator=mock_plan_generator, tool_manager={"search": DummyTool()})
        self.agent.perceive("Find AI models")

    def test_generate_plan(self):
        plan = self.agent.generate_plan("Test input")
        self.assertIsInstance(plan, list)
        self.assertGreaterEqual(len(plan), 1)

    def test_decide_next_action(self):
        action1 = self.agent.decide_next_action()
        self.assertEqual(action1["type"], "tool_use")
        action2 = self.agent.decide_next_action()
        self.assertEqual(action2["type"], "respond")

    def test_interrupt_and_replan(self):
        self.agent.interrupt_and_replan("New context")
        self.assertEqual(self.agent.current_action_index, 0)
        self.assertTrue(len(self.agent.active_plan) > 0)

if __name__ == "__main__":
    unittest.main()
