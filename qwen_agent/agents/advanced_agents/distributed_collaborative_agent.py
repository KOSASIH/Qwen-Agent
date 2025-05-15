"""
Distributed Collaborative Agent for Qwen-Agent framework
Coordinates multiple agents running distributedly,
exchanging knowledge and collaboratively solving complex problems.
"""

from typing import Any, Dict, List, Optional
from threading import Lock
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent

class DistributedCollaborativeAgent(BaseAdvancedAgent):
    def __init__(self, *args, peer_agents: Optional[List[BaseAdvancedAgent]] = None, **kwargs):
        """
        Initialize with a list of peer agents forming the collaborative network.

        Args:
            peer_agents: List of other agents participating in collaboration.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.peer_agents = peer_agents or []
        self.lock = Lock()
        self.shared_knowledge = {}

    def broadcast_knowledge(self, key: str, knowledge: Any) -> None:
        """
        Shares knowledge among all peer agents.

        Args:
            key: Knowledge key/identifier.
            knowledge: The knowledge content to share.
        """
        with self.lock:
            self.shared_knowledge[key] = knowledge
            self._log_debug(f"Broadcasted knowledge key: {key}")

    def receive_knowledge(self, key: str) -> Optional[Any]:
        """
        Retrieves shared knowledge by key.

        Args:
            key: Knowledge key to retrieve.

        Returns:
            The knowledge content or None if not found.
        """
        with self.lock:
            knowledge = self.shared_knowledge.get(key)
        self._log_debug(f"Received knowledge for key {key}: {knowledge}")
        return knowledge

    def coordinate_task(self, task: Dict[str, Any]) -> List[Any]:
        """
        Distributes parts of a task to peer agents and aggregates results.

        Args:
            task: The main task dict to coordinate.

        Returns:
            List of results from each peer agent.
        """
        results = []
        for agent in self.peer_agents:
            agent._log_debug(f"Coordinating task delegation to: {agent.__class__.__name__}")
            agent.perceive(task.get("content", ""))
            result = agent.decide_next_action()
            results.append(result)
        self._log_debug(f"Aggregated results from peers: {results}")
        return results

    def generate_plan(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Overrides plan generation to leverage distributed collaboration.

        Args:
            input_text: Input task or query.

        Returns:
            Coordination results merged as plan.
        """
        task = {"type": "collaborative_task", "content": input_text}
        results = self.coordinate_task(task)
        plan = [{"type": "collaboration_result", "content": r} for r in results]
        self._log_debug(f"Generated distributed collaboration plan: {plan}")
        return plan
