"""
Explainable AI Agent for Qwen-Agent framework
Generates traceable and human-understandable explanations
for decision-making and responses, enhancing transparency.
"""

from typing import Any, Dict, List
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class ExplainableAIAgent(BaseAdvancedAgent):
    def __init__(self, *args, explanation_generator=None, **kwargs):
        """
        Initialize with optional explanation generation module.

        Args:
            explanation_generator: Callable that produces explanations per decision.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.explanation_generator = explanation_generator
        self.explanations: List[str] = []

    def decide_next_action(self) -> Dict[str, Any]:
        """
        Override to produce explanation alongside the decision.

        Returns:
            Dict containing action and explanation.
        """
        action = super().decide_next_action()
        explanation = ""
        if self.explanation_generator:
            try:
                explanation = self.explanation_generator(action)
                self.explanations.append(explanation)
            except Exception as e:
                explanation = f"Explanation generation failed: {e}"
        
        # Incorporate explanation into the action dict
        action_with_explanation = {
            "action": action,
            "explanation": explanation
        }
        self._log_debug(f"Action with explanation: {action_with_explanation}")
        return action_with_explanation

    def get_explanations(self) -> List[str]:
        """
        Retrieve all stored explanations.

        Returns:
            List of explanation strings.
        """
        return self.explanations
