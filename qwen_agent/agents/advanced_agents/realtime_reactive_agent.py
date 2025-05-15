"""
Real-Time Reactive Agent for Qwen-Agent framework
Designed to process streaming data inputs rapidly,
manage immediate responses, and adapt behavior in time-sensitive contexts.
"""

from typing import Any, Dict, Optional
from qwen_agent.agents.advanced_agents.base_advanced_agent import BaseAdvancedAgent
import threading
import queue
import time

class RealTimeReactiveAgent(BaseAdvancedAgent):
    def __init__(self, *args, max_latency: float = 0.1, **kwargs):
        """
        Initialize with max latency constraints and streaming queue.

        Args:
            max_latency: Maximum allowed response time in seconds.
            *args, **kwargs: Passed to BaseAdvancedAgent.
        """
        super().__init__(*args, **kwargs)
        self.max_latency = max_latency
        self.input_queue = queue.Queue()
        self.running = False
        self.response_callback = None  # Optional callable for async responses

    def start(self) -> None:
        """
        Starts the background thread for processing streaming inputs.
        """
        if not self.running:
            self.running = True
            threading.Thread(target=self._processing_loop, daemon=True).start()
            self._log_debug("RealTimeReactiveAgent started processing loop.")

    def stop(self) -> None:
        """
        Stops the background processing thread.
        """
        self.running = False
        self._log_debug("RealTimeReactiveAgent stopped processing loop.")

    def enqueue_input(self, observation: str, modalities: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds new input for processing.

        Args:
            observation: Incoming input text or data.
            modalities: Optional multimodal data.
        """
        self.input_queue.put((observation, modalities))
        self._log_debug(f"Enqueued new input: {observation}")

    def _processing_loop(self) -> None:
        """
        Process inputs from the queue and respond respecting max latency.
        """
        while self.running:
            try:
                observation, modalities = self.input_queue.get(timeout=0.05)
                start = time.time()
                self.perceive(observation, modalities)
                action = self.decide_next_action()
                latency = time.time() - start
                self._log_debug(f"Processed input with latency {latency:.4f}s: {action}")

                if self.response_callback:
                    self.response_callback(action)
                # Enforce max latency (could log warnings or trigger fallback)
                if latency > self.max_latency:
                    self._log_debug(f"Warning: Response latency ({latency:.4f}s) exceeds max ({self.max_latency}s).")
            except queue.Empty:
                continue

