import unittest
from qwen_agent.agents.advanced_agents.distributed_collaborative_agent import DistributedCollaborativeAgent
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class DummyPeerAgent(BaseAdvancedAgent):
    def __init__(self):
        super().__init__()
    def decide_next_action(self):
        return {"type": "response", "content": "Peer agent response"}

class TestDistributedCollaborativeAgent(unittest.TestCase):
    def setUp(self):
        self.peer1 = DummyPeerAgent()
        self.peer2 = DummyPeerAgent()
        self.agent = DistributedCollaborativeAgent(peer_agents=[self.peer1, self.peer2])

    def test_broadcast_and_receive_knowledge(self):
        self.agent.broadcast_knowledge("key", "value")
        val = self.agent.receive_knowledge("key")
        self.assertEqual(val, "value")

    def test_coordinate_task_delegation(self):
        task = {"type": "task", "content": "Do something"}
        results = self.agent.coordinate_task(task)
        self.assertEqual(len(results), 2)

    def test_generate_plan_collaborates(self):
        plan = self.agent.generate_plan("Solve this problem")
        self.assertEqual(len(plan), 2)
        self.assertTrue(all(item["type"] == "collaboration_result" for item in plan))

if __name__ == "__main__":
    unittest.main()
