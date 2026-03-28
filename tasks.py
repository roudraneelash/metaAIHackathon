from __future__ import annotations

import json
from pathlib import Path

from models import EmailRecord


TASKS = {
    1: {
        "id": 1,
        "name": "Category Classification",
        "difficulty": "easy",
        "instructions": "Classify the email into the correct business category.",
        "allowed_fields": ["category"],
    },
    2: {
        "id": 2,
        "name": "Category And Priority",
        "difficulty": "medium",
        "instructions": (
            "Classify the email category and estimate the correct priority level."
        ),
        "allowed_fields": ["category", "priority"],
    },
    3: {
        "id": 3,
        "name": "Full Triage Routing",
        "difficulty": "hard",
        "instructions": (
            "Classify the email category, priority, owner department, "
            "and the best response action."
        ),
        "allowed_fields": ["category", "priority", "department", "response_action"],
    },
}


def load_dataset() -> list[EmailRecord]:
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "dataset.json"
    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [EmailRecord.model_validate(r) for r in raw]


def get_task_definition(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return TASKS[task_id]
