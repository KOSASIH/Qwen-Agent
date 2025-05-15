"""
Self-Improving Agent for Qwen-Agent framework
Implements continuous self-monitoring, performance evaluation,
and iterative self-enhancement of reasoning and action quality.
"""

from typing import Any, Dict, List
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class SelfImprovingAgent(BaseAdvancedAgent):
    def __init__(self, *args, performance_tracker=None, improvement_strategy=None, **kwargs):
        """
        Initialize with optional performance tracker and improvement modules.

        Args:
            performance_tracker: Callable or module to track agent performance.
            improvement_strategy: Callable/module implementing learning and improvement.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.performance_tracker = performance_tracker
        self.improvement_strategy = improvement_strategy
        self.history: List[Dict[str, Any]] = []

    def decide_next_action(self) -> Dict[str, Any]:
        """
        Decide next action and log the action for self-evaluation.

        Returns:
            Dict describing the chosen action.
        """
        action = super().decide_next_action()
        self._log_debug(f"Action decided: {action}")

        # Track for performance evaluation
        if self.performance_tracker:
            self.performance_tracker.record_action(action)
        self.history.append(action)

        return action

    def evaluate_performance(self) -> None:
        """
        Evaluate past actions and identify areas for improvement.
        """
        if self.performance_tracker:
            eval_result = self.performance_tracker.evaluate(self.history)
            self._log_debug(f"Performance evaluation result: {eval_result}")
            self.apply_improvements(eval_result)

    def apply_improvements(self, evaluation_data: Any) -> None:
        """
        Apply improvements to agent's logic or parameters based on evaluation.

        Args:
            evaluation_data: Data or recommendations from the evaluation.
        """
        if self.improvement_strategy:
            self.improvement_strategy.apply(evaluation_data)
            self._log_debug("Applied improvements based on evaluation.")

