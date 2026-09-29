from dataclasses import dataclass


@dataclass
class TrainerContext:
    """Per-request context the agent passes to every tool; never exposed to the LLM."""
    client_id: str
