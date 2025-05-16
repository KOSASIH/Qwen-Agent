import unittest
from qwen_agent.agents.advanced_agents.multimodal_agent import MultimodalAgent

def dummy_image_processor(image):
    return "processed_image"

class DummyMemoryManager:
    def __init__(self):
        self.mem = {}
    def add_memory(self, key, value):
        self.mem[key] = value
    def retrieve(self, key):
        return self.mem.get(key, None)

class TestMultimodalAgent(unittest.TestCase):
    def setUp(self):
        self.memory_manager = DummyMemoryManager()
        self.agent = MultimodalAgent(memory_manager=self.memory_manager, modality_processors={"image": dummy_image_processor})

    def test_perceive_processes_modalities(self):
        self.agent.perceive("Text input", modalities={"image": "raw_image_data"})
        stored = self.memory_manager.retrieve("processed_image")
        self.assertEqual(stored, "processed_image")

    def test_generate_plan_includes_multimodal(self):
        self.agent.perceive("Analyze this", modalities={"image": "image_data"})
        plan = self.agent.generate_plan("Analyze this")
        self.assertIsInstance(plan, list)
        found_multimodal = any(act.get("type") == "use_multimodal_data" for act in plan)
        self.assertTrue(found_multimodal)

    def test_respond_multimodal(self):
        response = self.agent.respond_multimodal("Response text", modalities={"audio": "audio_data"})
        self.assertIn("modalities", response)
        self.assertEqual(response["text"], "Response text")

if __name__ == "__main__":
    unittest.main()
