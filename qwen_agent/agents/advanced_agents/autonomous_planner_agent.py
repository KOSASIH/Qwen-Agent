"""
Autonomous Planner Agent for Qwen-Agent framework
This agent extends BaseAdvancedAgent to provide powerful multi-step
autonomous planning capabilities with adaptive execution and dynamic tool usage.
"""

from typing import Any, Dict, List, Optional
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class AutonomousPlannerAgent(BaseAdvancedAgent):
    def __init__(self, *args, plan_generator=None, **kwargs):
        """
        Initialize AutonomousPlannerAgent with optional plan generation module.

        Args:
            plan_generator: Callable or module responsible for generating plans.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.plan_generator = plan_generator

        # Stores the current active plan: list of steps/actions
        self.active_plan: List[Dict[str, Any]] = []
        self.current_action_index: int = 0

    def generate_plan(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Generate a detailed multi-step plan, overriding base method.

        If a plan_generator callable is provided, use it; otherwise fallback.

        Args:
            input_text: The input prompt or goal.

        Returns:
            List of action steps (dicts).
        """
        if self.plan_generator:
            try:
                plan = self.plan_generator(input_text)
                self._log_debug(f"Plan generated by external generator: {plan}")
                return plan
            except Exception as e:
                self._log_debug(f"Plan generator failed: {e}")

        # Fallback simple plan for demonstration
        fallback_plan = [
            {"type": "analyze", "content": input_text},
            {"type": "tool_use", "tool_name": "search_engine", "parameters": {"query": input_text}},
            {"type": "respond", "content": "Here is the information I found."}
        ]
        self._log_debug("Fallback plan used")
        return fallback_plan

    def decide_next_action(self) -> Dict[str, Any]:
        """
        Decide and return the next action to execute from the active plan.

        Manages plan progression and automatic regeneration if exhausted.

        Returns:
            Action dict to execute next.
        """
        self._log_debug("AutonomousPlannerAgent deciding next action.")

        if not self.active_plan or self.current_action_index >= len(self.active_plan):
            # Generate a fresh plan based on latest observation
            self.active_plan = self.generate_plan(self.last_observation)
            self.current_action_index = 0
            self._log_debug(f"New active plan set: {self.active_plan}")

        # Fetch the current action
        action = self.active_plan[self.current_action_index]
        self.current_action_index += 1

        self._log_debug(f"Next action: {action}")

        # If action requires a tool, invoke it automatically here or elsewhere in framework
        if action.get("type") == "tool_use":
            tool_name = action.get("tool_name")
            params = action.get("parameters", {})
            tool_result = self.invoke_tool(tool_name, params)
            # You might store tool_result or adapt plan based on this result
            self.remember(f"tool_result_{tool_name}", tool_result)
            self._log_debug(f"Tool result stored for {tool_name}")

        return action

    def interrupt_and_replan(self, new_context: str) -> None:
        """
        Interrupt the current plan and regenerate it based on new context.

        Args:
            new_context: New information requiring replanning.
        """
        self._log_debug(f"Interrupting plan due to new context: {new_context}")
        self.active_plan = self.generate_plan(new_context)
        self.current_action_index = 0

