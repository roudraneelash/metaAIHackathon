#!/usr/bin/env python3
"""
Inference script for the Email Triage OpenEnv environment.

Uses the competition-mandated environment variables:
  API_BASE_URL  – LLM provider base URL
  MODEL_NAME    – model identifier
  HF_TOKEN      – authentication token

Can run against a local server (default http://localhost:8000) or a
remote HuggingFace Space URL passed via ENV_URL.

Uses the WebSocket-based EnvClient for multi-step episodes.
"""
from __future__ import annotations

import json
import os
import sys

import httpx
from openai import OpenAI

from client import EmailTriageEnvClient
from models import EmailTriageAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SEED = 42
TASKS = [1, 2, 3]

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

llm_client: OpenAI | None = None

if MODEL_NAME and HF_TOKEN:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


SYSTEM_PROMPT = """\
You are an expert email triage agent. Given a business email, you must produce a JSON object with the requested fields.

Valid values:
- category: billing, account_access, technical_support, sales, spam, general, compliance, onboarding, feedback
- priority: critical, high, medium, low
- department: finance, support, engineering, sales, operations, compliance, hr
- response_action: respond, escalate, forward, ignore, acknowledge

Return ONLY valid JSON with the requested fields. No markdown, no explanation."""


def call_llm(email: dict, allowed_fields: list[str], instructions: str) -> dict:
    assert llm_client is not None, "LLM client not configured"

    user_msg = (
        f"Instructions: {instructions}\n\n"
        f"Allowed fields: {', '.join(allowed_fields)}\n\n"
        f"Subject: {email['subject']}\n"
        f"From: {email['sender']}\n"
        f"Body: {email['body']}\n\n"
        f"Respond with JSON containing ONLY these fields: {', '.join(allowed_fields)}"
    )

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    text = response.choices[0].message.content or "{}"
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM needed)
# ---------------------------------------------------------------------------

KEYWORD_CATEGORIES = {
    "invoice": "billing",
    "charge": "billing",
    "refund": "billing",
    "payment": "billing",
    "billing": "billing",
    "sign in": "account_access",
    "login": "account_access",
    "password": "account_access",
    "locked": "account_access",
    "2fa": "account_access",
    "bug": "technical_support",
    "error": "technical_support",
    "exception": "technical_support",
    "crash": "technical_support",
    "production": "technical_support",
    "pricing": "sales",
    "quote": "sales",
    "demo": "sales",
    "enterprise": "sales",
    "rollout": "sales",
    "spam": "spam",
    "click now": "spam",
    "guaranteed": "spam",
    "unsubscribe": "spam",
    "compliance": "compliance",
    "regulation": "compliance",
    "gdpr": "compliance",
    "audit": "compliance",
    "onboarding": "onboarding",
    "welcome": "onboarding",
    "getting started": "onboarding",
    "feedback": "feedback",
    "suggestion": "feedback",
    "improve": "feedback",
}

CATEGORY_TO_DEPT = {
    "billing": "finance",
    "account_access": "support",
    "technical_support": "engineering",
    "sales": "sales",
    "spam": "operations",
    "general": "operations",
    "compliance": "compliance",
    "onboarding": "support",
    "feedback": "operations",
}

CATEGORY_TO_ACTION = {
    "billing": "respond",
    "account_access": "respond",
    "technical_support": "escalate",
    "sales": "forward",
    "spam": "ignore",
    "general": "acknowledge",
    "compliance": "escalate",
    "onboarding": "respond",
    "feedback": "acknowledge",
}


def heuristic_action(email: dict, allowed_fields: list[str]) -> dict:
    text = (email.get("subject", "") + " " + email.get("body", "")).lower()

    category = "general"
    for kw, cat in KEYWORD_CATEGORIES.items():
        if kw in text:
            category = cat
            break

    priority = "medium"
    if any(w in text for w in ["urgent", "critical", "blocking", "asap", "immediately"]):
        priority = "critical"
    elif any(w in text for w in ["important", "high priority", "revenue"]):
        priority = "high"
    elif any(w in text for w in ["low", "whenever", "no rush"]):
        priority = "low"

    result: dict = {}
    if "category" in allowed_fields:
        result["category"] = category
    if "priority" in allowed_fields:
        result["priority"] = priority
    if "department" in allowed_fields:
        result["department"] = CATEGORY_TO_DEPT.get(category, "operations")
    if "response_action" in allowed_fields:
        result["response_action"] = CATEGORY_TO_ACTION.get(category, "acknowledge")
    return result


# ---------------------------------------------------------------------------
# Main loop using WebSocket client for multi-step episodes
# ---------------------------------------------------------------------------

def run():
    # Quick HTTP health check
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)
    health = http.get("/health")
    health.raise_for_status()
    print(f"Connected to {ENV_URL}: {health.json()}")

    tasks_resp = http.get("/tasks")
    tasks_resp.raise_for_status()
    available_tasks = {t["id"]: t for t in tasks_resp.json()["tasks"]}
    print(f"Available tasks: {[t['name'] for t in available_tasks.values()]}")
    http.close()

    all_scores: dict[int, list[float]] = {}

    for task_id in TASKS:
        if task_id not in available_tasks:
            print(f"Task {task_id} not available, skipping")
            continue

        task = available_tasks[task_id]
        print(f"\n--- Task {task_id}: {task['name']} ({task['difficulty']}) ---")

        # Use sync WebSocket client for multi-step episode
        sync_client = EmailTriageEnvClient(base_url=ENV_URL).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=task_id)
            obs = result.observation

            task_scores: list[float] = []
            step_num = 0

            while not result.done:
                email = obs.current_email
                if email is None:
                    break

                allowed = obs.allowed_fields
                instructions = obs.instructions

                if llm_client is not None:
                    action_dict = call_llm(email, allowed, instructions)
                else:
                    action_dict = heuristic_action(email, allowed)

                action = EmailTriageAction(**action_dict)
                result = sync_client.step(action)
                obs = result.observation

                step_num += 1
                print(f"  Step {step_num}: reward={result.reward} done={result.done}")

                if result.reward is not None:
                    task_scores.append(result.reward)

        all_scores[task_id] = task_scores
        final = task_scores[-1] if task_scores else 0.0
        print(f"  Task {task_id} final reward: {final:.4f}")

    # Summary
    print("\n=== RESULTS ===")
    overall = []
    for tid in TASKS:
        if tid in all_scores:
            scores = all_scores[tid]
            avg = sum(scores) / len(scores) if scores else 0.0
            overall.append(avg)
            print(f"Task {tid}: avg_score={avg:.4f} ({len(scores)} steps)")
    if overall:
        print(f"Overall: {sum(overall) / len(overall):.4f}")


if __name__ == "__main__":
    run()
