"""
Multimodal Agent for Qwen-Agent framework
Extends the BaseAdvancedAgent to handle multimodal inputs and outputs:
text, image, audio, video, with flexible processing pipelines
and reasoning across modalities.
"""

from typing import Any, Dict, Optional
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class MultimodalAgent(BaseAdvancedAgent):
    def __init__(self, *args, modality_processors: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the MultimodalAgent.

        Args:
            modality_processors: Dict of callable processors keyed by modality name.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.modality_processors = modality_processors or {}

    def perceive(self, observation: str, modalities: Optional[Dict[str, Any]] = None) -> None:
        """
        Processes multimodal input including text and other modalities.

        Args:
            observation: Primary textual input.
            modalities: Dict containing multimodal data e.g., images, audio, video.
        """
        # Process each modality using the configured processors.
        if modalities:
            for modality, data in modalities.items():
                processor = self.modality_processors.get(modality)
                if processor:
                    processed = processor(data)
                    self._log_debug(f"Processed modality '{modality}': {processed}")
                    # Optionally store processed result in memory
                    if self.memory_manager:
                        self.memory_manager.add_memory(f"processed_{modality}", processed)

        # Always call base perceive for text
        super().perceive(observation, modalities)

    def generate_plan(self, input_text: str) -> Any:
        """
        Generate plan taking multimodal processed data and text into account.

        Args:
            input_text: Text input prompt.

        Returns:
            List of action dicts.
        """
        processed_images = self.memory_manager.retrieve("processed_image") if self.memory_manager else None
        # Use the combined info to create a plan (placeholder logic)
        plan = [
            {"type": "analyze", "content": input_text},
            {"type": "use_multimodal_data", "data": processed_images},
            {"type": "respond", "content": "Analysis complete with multimodal understanding."}
        ]
        self._log_debug(f"Generated multimodal plan: {plan}")
        return plan

    def respond_multimodal(self, content: str, modalities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare a response with possible multimodal content.

        Args:
            content: Textual response.
            modalities: Optional dict of response modalities, e.g. images, audio.

        Returns:
            Dict representing multimodal response.
        """
        response = {"type": "response", "text": content}
        if modalities:
            response["modalities"] = modalities
        self._log_debug(f"Prepared multimodal response: {response}")
        return response

