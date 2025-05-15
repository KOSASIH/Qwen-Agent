"""
Multi-Agent Coordinator for Qwen-Agent framework
Coordinates multiple specialized agents by delegating sub-tasks
and orchestrating collaborative workflows to solve complex problems.
"""

from typing import Any, Dict, List, Optional
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class MultiAgentCoordinator(BaseAdvancedAgent):
    def __init__(self, *args, agents: Optional[Dict[str, BaseAdvancedAgent]] = None, **kwargs):
        """
        Initialize with a dictionary of specialized agents.

        Args:
            agents: Dict with keys as agent names and values as agent instances.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.agents = agents or {}

    def delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate a sub-task to an appropriate agent.

        Args:
            task: Dict representing the sub-task with type and content.

        Returns:
            Result dict from delegated agent.
        """
        task_type = task.get("type")
        agent = self.select_agent(task_type)
        if agent:
            self._log_debug(f"Delegating task {task} to agent {agent.__class__.__name__}")
            # Feed task content as observation and get next action
            agent.perceive(task.get("content", ""))
            result = agent.decide_next_action()
            return result
        else:
            self._log_debug(f"No agent found to handle task type: {task_type}")
            return {"error": f"No suitable agent for task type {task_type}"}

    def select_agent(self, task_type: str) -> Optional[BaseAdvancedAgent]:
        """
        Select an agent by task type.

        Args:
            task_type: The task type string.

        Returns:
            Agent instance, or None if no match.
        """
        # Simple selection logic (could be extended)
        agent = self.agents.get(task_type)
        if agent:
            return agent

        # Fallback to any agent that declares support for the task
        for agent_instance in self.agents.values():
            if hasattr(agent_instance, "supported_task_types") and task_type in agent_instance.supported_task_types:
                return agent_instance

        return None

    def generate_plan(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Generate a broad coordination plan that decomposes into sub-tasks,
        delegates them, and aggregates results.

        Args:
            input_text: User prompt or main task.

        Returns:
            List of coordinated action dicts.
        """
        self._log_debug(f"Generating coordination plan for: {input_text}")

        # Simple decomposition example: split input into tasks by sentence
        tasks = [{"type": "analyze", "content": sentence.strip()} 
                 for sentence in input_text.split(".") if sentence.strip()]

        plan = []
        for task in tasks:
            result = self.delegate_task(task)
            plan.append({"type": "delegated_result", "content": result})

        return plan
