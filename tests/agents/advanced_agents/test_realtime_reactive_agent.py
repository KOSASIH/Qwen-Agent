import unittest
import time
from qwen_agent.agents.advanced_agents.realtime_reactive_agent import RealTimeReactiveAgent

class TestRealTimeReactiveAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RealTimeReactiveAgent()

    def test_enqueue_and_process(self):
        results = []
        def callback(action):
            results.append(action)

        self.agent.response_callback = callback
        self.agent.start()
        self.agent.enqueue_input("Hello real-time")
        time.sleep(0.2)  # Allow processing to occur
        self.agent.stop()
        self.assertTrue(len(results) > 0)

if __name__ == "__main__":
    unittest.main()
