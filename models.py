from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class EmailRecord(BaseModel):
    email_id: str
    subject: str
    sender: str
    body: str
    category: str
    priority: str
    department: str
    response_action: str
    ambiguity_note: Optional[str] = None
    related_email_id: Optional[str] = None


class EmailTriageAction(Action):
    category: Optional[str] = None
    priority: Optional[str] = None
    department: Optional[str] = None
    response_action: Optional[str] = None


class EmailTriageObservation(Observation):
    task_id: int = 0
    task_name: str = ""
    instructions: str = ""
    allowed_fields: list[str] = Field(default_factory=list)
    current_email: Optional[dict[str, str]] = None
    inbox_size: int = 0
    emails_remaining: int = 0
    emails_processed: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)


class EmailTriageState(State):
    current_task_id: Optional[int] = None
    seed: Optional[int] = None
    inbox_email_ids: list[str] = Field(default_factory=list)
    current_email_index: int = 0
    per_email_scores: list[float] = Field(default_factory=list)
    total_reward: float = 0.0
