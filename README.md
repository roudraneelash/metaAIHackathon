# Email Triage OpenEnv

> **Meta PyTorch OpenEnv Hackathon — Round 1 Submission**
> **Team Hijibiji** — Roudraneel, Rijul, Tirthajoti

A deterministic, multi-step email triage environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent receives a randomized inbox of business emails and must classify, prioritize, route, and decide the response action for each one — simulating a real enterprise email triage workflow.

---

## Why Email Triage?

Email triage is a genuine enterprise task — every organization with a shared inbox (support@, info@, billing@) needs someone (or something) to read each message, decide what it's about, how urgent it is, who should handle it, and what to do next. Companies like Zendesk, Freshdesk, and ServiceNow build products around this exact workflow.

This environment fills a gap in the OpenEnv ecosystem: **no existing environment models email classification and routing as a multi-step, multi-field decision problem** with partial-credit scoring.

---

## Architecture

```
+-------------+     WebSocket      +------------------+
| inference.py | <---------------> |  server/app.py   |
| (AI agent)   |                   |  (FastAPI server) |
+-------------+                    +--------+---------+
       |                                    |
       | uses                        calls  |
       v                                    v
+-------------+                    +------------------+
|  client.py  |                    |  environment.py  |
| (WebSocket  |                    |  (core engine)   |
|  client)    |                    +---+----+----+----+
+-------------+                        |    |    |
                                       v    v    v
                               grader  reward  tasks
                               .py     .py     .py
                                              |
                                              v
                                        dataset.json
                                        (45 emails)
```

**Key architectural detail:** The OpenEnv `create_app()` HTTP endpoints (`/reset`, `/step`) are stateless — each request creates a new environment instance. Multi-step episodes (processing 3–5 emails in sequence) **require the WebSocket** endpoint (`/ws`), which maintains session state. The `client.py` and `inference.py` use the WebSocket-based `EnvClient` for this reason.

---

## Tasks

| ID | Name | Difficulty | Fields Required | Description |
|----|------|------------|-----------------|-------------|
| 1 | Category Classification | Easy | `category` | Classify the email into one of 9 business categories |
| 2 | Category + Priority | Medium | `category`, `priority` | Classify category and estimate urgency level |
| 3 | Full Triage Routing | Hard | `category`, `priority`, `department`, `response_action` | Complete triage: classify, prioritize, route to department, and decide response |

**Why Task 3 is genuinely hard:**
- Some emails are **deliberately ambiguous** — e.g., subject says "billing issue" but the body describes a technical problem
- Some emails **reference previous threads** — requiring contextual reasoning
- The agent must get **four fields** correct simultaneously, with weighted scoring

---

## Action Space

The agent submits an `EmailTriageAction` for each email. Only the fields relevant to the current task are scored; others are ignored.

```json
{
  "category": "billing | account_access | technical_support | sales | spam | general | compliance | onboarding | feedback",
  "priority": "critical | high | medium | low",
  "department": "finance | support | engineering | sales | operations | compliance | hr",
  "response_action": "respond | escalate | forward | ignore | acknowledge"
}
```

All fields are optional strings. The agent should only fill in the fields listed in `allowed_fields` from the observation.

## Observation Space

Each observation (`EmailTriageObservation`) the agent receives contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Current task ID (1, 2, or 3) |
| `task_name` | str | Human-readable task name |
| `instructions` | str | What the agent should do |
| `allowed_fields` | list[str] | Which action fields to fill in |
| `current_email` | dict | The email to triage: `email_id`, `subject`, `sender`, `body` |
| `inbox_size` | int | Total emails in this episode's inbox |
| `emails_remaining` | int | How many emails are left |
| `emails_processed` | int | How many have been processed |
| `history` | list[dict] | Scores from previous steps |
| `done` | bool | Whether the episode is complete |
| `reward` | float \| null | Reward for the last action (trajectory reward on final step) |

**Note:** The observation intentionally strips ground-truth labels from emails — the agent only sees `email_id`, `subject`, `sender`, and `body`.

## State

The internal `EmailTriageState` tracks:

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | str | Unique episode identifier |
| `step_count` | int | Actions taken so far |
| `current_task_id` | int | Active task |
| `seed` | int | RNG seed for reproducibility |
| `inbox_email_ids` | list[str] | Email IDs in the current inbox |
| `current_email_index` | int | Index of the current email |
| `per_email_scores` | list[float] | Score for each processed email |
| `total_reward` | float | Final trajectory reward |

---

## Grading Methodology

All grading is **deterministic** — same input always produces the same score. Scores range continuously from 0.0 to 1.0.

### Per-Field Scoring

**Category** — Exact match or partial credit via similarity matrix:
| Predicted vs Actual | Score |
|---|---|
| Exact match | 1.0 |
| billing ↔ payments | 0.7 |
| technical_support ↔ account_access | 0.5 |
| billing ↔ finance | 0.4 |
| onboarding ↔ account_access | 0.4 |
| general ↔ feedback | 0.3 |
| compliance ↔ billing | 0.2 |
| spam ↔ sales | 0.1 |
| No match in similarity table | 0.0 |

**Priority** — Exact match or proximity score:
| Distance | Example | Score |
|---|---|---|
| Exact | critical → critical | 1.0 |
| 1 level | critical ↔ high | 0.6 |
| 1 level | high ↔ medium | 0.5 |
| 1 level | medium ↔ low | 0.4 |
| 2 levels | critical ↔ medium | 0.3 |
| 2 levels | high ↔ low | 0.2 |
| 3 levels | critical ↔ low | 0.1 |

**Department & Response Action** — Exact match only: 1.0 or 0.0.

### Per-Task Weighting

| Task | Category | Priority | Department | Response Action |
|------|----------|----------|------------|-----------------|
| 1 (Easy) | 100% | — | — | — |
| 2 (Medium) | 60% | 40% | — | — |
| 3 (Hard) | 35% | 20% | 25% | 20% |

### Trajectory Reward

At the end of an episode (all emails processed), the trajectory reward is:

```
trajectory_reward = average(per_email_scores) - 0.03 * max(0, steps_taken - inbox_size)
```

The small penalty discourages agents from taking unnecessary extra steps. The final reward is clamped to [0.0, 1.0].

---

## Dataset

`data/dataset.json` contains **45 labeled business emails** covering:

- **9 categories:** billing, account_access, technical_support, sales, spam, general, compliance, onboarding, feedback
- **4 priority levels:** critical, high, medium, low
- **7 departments:** finance, support, engineering, sales, operations, compliance, hr
- **5 response actions:** respond, escalate, forward, ignore, acknowledge

Each email includes ground-truth labels for all fields. Select emails include:
- `ambiguity_note` — explains why the email could be misclassified
- `related_email_id` — references another email in a thread (tests contextual reasoning)

Each episode randomly samples 3–5 emails from the dataset (seeded for reproducibility).

---

## Project Structure

```
MetaHackathonPrep/
├── server/
│   ├── __init__.py         # Package marker
│   ├── app.py              # FastAPI server via create_app() + /tasks endpoint
│   ├── environment.py      # EmailTriageEnvironment (reset/step/state)
│   ├── grader.py           # Deterministic scoring with similarity matrices
│   ├── reward.py           # Step and trajectory reward computation
│   ├── tasks.py            # Task definitions + dataset loader
│   └── Dockerfile          # Docker image for HuggingFace Spaces (port 7860)
├── data/
│   └── dataset.json        # 45 labeled business emails
├── models.py               # EmailTriageAction, Observation, State (extend OpenEnv base)
├── client.py               # WebSocket EnvClient subclass for multi-step episodes
├── inference.py             # LLM inference script (mandatory) + heuristic fallback
├── openenv.yaml            # OpenEnv environment metadata
├── pyproject.toml          # pip-installable package configuration
├── requirements.txt        # Python dependencies
├── KNOWLEDGE.md            # Beginner-friendly full project explanation
├── PLAN.md                 # Master implementation plan
└── README.md               # This file
```

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- pip

### Local Installation

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or via Python directly:

```bash
python -m server.app
```

### Verify

```bash
curl http://localhost:8000/health
# {"status": "healthy"}

curl http://localhost:8000/tasks
# {"tasks": [{"id": 1, "name": "Category Classification", ...}, ...]}
```

---

## Running Inference

### With LLM (competition mode)

Set the three mandated environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

The script connects to the server via WebSocket, runs all 3 tasks (seed=42), and prints per-task and overall scores.

### Without LLM (heuristic baseline)

Simply run without setting the environment variables:

```bash
python inference.py
```

The script falls back to a keyword-matching heuristic — useful for development and testing without API costs.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | For LLM mode | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | For LLM mode | Model identifier |
| `HF_TOKEN` | For LLM mode | HuggingFace / API authentication token |
| `ENV_URL` | No | Server URL (default: `http://localhost:8000`) |

---

## Docker

### Build and Run

```bash
docker build -f server/Dockerfile -t email-triage .
docker run -p 7860:7860 email-triage
```

### Run Inference Against Docker Container

```bash
ENV_URL=http://localhost:7860 python inference.py
```

### HuggingFace Spaces

The Dockerfile exposes port 7860 (required by HF Spaces). Deploy by pushing the repository to a Docker-type HuggingFace Space.

---

## API Endpoints

All endpoints are auto-generated by OpenEnv's `create_app()`, plus one custom route.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status": "healthy"}` |
| POST | `/reset` | Start a new episode — accepts `{"seed": int, "task_id": int}` |
| POST | `/step` | Submit an action — accepts `{"action": {...}}` |
| GET | `/state` | Get current environment state |
| WebSocket | `/ws` | Persistent connection for multi-step episodes |
| GET | `/tasks` | List available tasks with descriptions *(custom)* |
| GET | `/docs` | Auto-generated OpenAPI documentation |
| GET | `/schema` | JSON schema for action/observation models |
| GET | `/metadata` | Environment metadata |

**Important:** Use the WebSocket `/ws` endpoint (via `client.py`) for multi-step episodes. The HTTP `/reset` and `/step` endpoints are stateless and create a new environment instance per request.

---

## Episode Flow

1. **Connect** via WebSocket to the server
2. **Reset** with `{"seed": 42, "task_id": 3}` — the environment samples a random inbox of 3–5 emails
3. **Receive** first observation — contains the first email to triage, task instructions, and allowed fields
4. **Step** — submit an `EmailTriageAction` with your classification
5. **Receive** next observation — the grader scores your action (0.0–1.0), and the next email is shown
6. **Repeat** steps 4–5 until all emails are processed (`done=true`)
7. **Final reward** — the last observation contains the trajectory reward (average of all email scores, minus step penalty)

Example single-task flow:
```
reset(seed=42, task_id=1)  →  Observation(email_1, inbox_size=4, emails_remaining=4)
step(category="billing")   →  Observation(email_2, reward=1.0, emails_remaining=3)
step(category="spam")      →  Observation(email_3, reward=0.7, emails_remaining=2)
step(category="sales")     →  Observation(email_4, reward=1.0, emails_remaining=1)
step(category="compliance")→  Observation(done=true, reward=0.925)  ← trajectory reward
```

---

## Baseline Scores

Heuristic keyword-matching agent (no LLM, seed=42):

```
Task 1 (Category Classification — easy):    1.0000
Task 2 (Category + Priority — medium):      0.8960
Task 3 (Full Triage Routing — hard):        0.9480
Overall:                                     0.9480
```

These scores are fully reproducible with the same seed.

---

## OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| Extends `Action`, `Observation`, `State` base types | Yes — `models.py` |
| Implements `reset()`, `step()`, `state()` | Yes — `server/environment.py` |
| `openenv.yaml` with metadata | Yes |
| 3+ tasks with difficulty progression | Yes — easy/medium/hard |
| Deterministic graders (0.0–1.0) | Yes — `server/grader.py` |
| Varied score output | Yes — tested range 0.04 to 1.0 |
| Meaningful partial-credit rewards | Yes — similarity matrices + trajectory averaging |
| Baseline `inference.py` with env vars | Yes — reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` |
| Working Dockerfile | Yes — `server/Dockerfile`, port 7860 |
| pip-installable | Yes — `pyproject.toml` |

---

## Team

**Team Hijibiji** — Roudraneel, Rijul, Tirthajoti

Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv) — Round 1.