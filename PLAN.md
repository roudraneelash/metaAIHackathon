# Email Triage OpenEnv — Master Plan

## Team & Timeline
- **Team Hijibiji**: Roudraneel (lead), Rijul, Tirthajoti
- **Deadline**: April 7, 2026, 11:59 PM IST
- **Today**: March 28 — submission window just opened
- **Decision**: Start fresh following official OpenEnv standards exactly

---

## 1. COMPETITION REQUIREMENTS CHECKLIST

### Pass/Fail Gate (Phase 1 — Automated)
- [ ] HF Space deploys and returns 200 on GET, responds to reset()
- [ ] openenv.yaml valid, typed models, step()/reset()/state() endpoints
- [ ] Dockerfile builds: `docker build && docker run` works
- [ ] Baseline inference script (`inference.py`) runs without error, produces scores
- [ ] 3+ tasks enumerated, each grader scores in 0.0–1.0 range

### Scored Criteria
| Criterion | Weight | Our Target |
|---|---|---|
| Real-world utility | 30% | Email triage — genuine enterprise task, fills RL gap |
| Task & grader quality | 25% | 3 tasks, deterministic graders, hard task challenges frontier |
| Environment design | 20% | Multi-step inbox, shaped rewards, clean state via openenv base |
| Code quality & spec | 15% | openenv validate passes, typed models, Docker works |
| Creativity & novelty | 10% | Thread-aware triage, ambiguity handling, context-dependent priority |

### Disqualification Triggers
- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

---

## 2. OPENENV FRAMEWORK — VERIFIED TECHNICAL SPEC

### Base Types (from `openenv.core.env_server.types` — Pydantic BaseModel)

**Action**:
- Inherits: `metadata: Dict[str, Any]`
- Config: `extra="forbid"`, `validate_assignment=True`

**Observation**:
- Inherits: `done: bool = False`, `reward: float|None = None`, `metadata: Dict[str, Any]`
- Config: `extra="forbid"`, `validate_assignment=True`

**State**:
- Inherits: `episode_id: Optional[str] = None`, `step_count: int = 0`
- Config: `extra="allow"`, `validate_assignment=True`

### Environment Base Class (from `openenv.core.env_server.interfaces`)
```
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    def __init__(self, transform=None, rubric=None)
    
    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT
    
    @abstractmethod
    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT
    
    @property
    @abstractmethod
    def state(self) -> StateT
    
    def get_metadata(self) -> EnvironmentMetadata
    def close(self) -> None
```

Key: `step()` returns an **Observation** (with .done and .reward), NOT a separate StepResponse.

### Server Creation
```
from openenv.core.env_server import create_app
app = create_app(env, ActionClass, ObservationClass)
```
Auto-provides: POST /reset, POST /step, GET /state, GET /health, WebSocket /ws, GET /docs, GET /web

### Client
```
from openenv.core import EnvClient, StepResult

class MyEnv(EnvClient[ActT, ObsT, StateT]):
    def _step_payload(self, action) -> dict
    def _parse_result(self, payload) -> StepResult[ObsT]
    def _parse_state(self, payload) -> StateT
```
Usage: `async with MyEnv(base_url=...) as client:` or `.sync()` wrapper.
`StepResult` has: `.observation`, `.reward`, `.done`

### Health Endpoint
Returns: `{"status": "healthy"}` (not "ok")

### Official Scaffold (from `openenv init`)
```
my_env/
├── server/
│   ├── app.py            # create_app() + custom routes
│   ├── environment.py    # Environment subclass
│   └── Dockerfile
├── models.py             # Action/Observation/State subclasses
├── client.py             # EnvClient subclass
├── openenv.yaml
├── pyproject.toml        # pip-installable
└── README.md
```

### Port
- Dev: 8000
- HF Spaces Docker: 7860

### Install
```
pip install git+https://github.com/meta-pytorch/OpenEnv.git
```

---

## 3. INFERENCE SCRIPT REQUIREMENTS

### Mandatory
- File: `inference.py` at project root
- Env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Must use `OpenAI` client: `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
- LLM calls via `client.chat.completions.create(model=MODEL_NAME, ...)`
- Runtime: <20 min on 2 vCPU / 8 GB RAM

### Pattern (from sample BrowserGym script)
1. Create OpenAI client with env vars
2. Instantiate environment (or connect to HF Space)
3. For each task:
   a. env.reset(task_id=..., seed=...)
   b. Build prompt from observation
   c. Call LLM → parse response → construct action
   d. env.step(action)
   e. Collect reward
4. Print reproducible scores

---

## 4. TARGET PROJECT STRUCTURE (Fresh Start)

```
email_triage_env/
├── server/
│   ├── __init__.py
│   ├── app.py              # create_app(env, Action, Obs) + custom /tasks, /grader
│   ├── environment.py      # EmailTriageEnvironment(Environment)
│   ├── grader.py           # grade_action() — deterministic scoring
│   ├── reward.py           # compute_reward() — per-step + trajectory
│   ├── tasks.py            # TASKS dict + load_dataset()
│   └── Dockerfile          # python:3.11-slim, port 7860
├── models.py               # EmailTriageAction(Action), Observation(Observation), State(State)
├── client.py               # EmailTriageEnv(EnvClient)
├── inference.py            # Mandatory LLM inference script
├── data/
│   └── dataset.json        # 40+ labeled business emails
├── openenv.yaml            # Metadata
├── pyproject.toml           # pip-installable from HF Space
├── requirements.txt        # fastapi, pydantic, uvicorn, openai
└── README.md               # Full documentation
```

Discarded from old scaffold: `env/`, `api/`, `baseline/`, root `Dockerfile`

---

## 5. ENVIRONMENT DESIGN

### Domain: Email Triage
An inbox of business emails that an agent must classify, prioritize, route to the correct department, and decide on a response action. This models a genuine enterprise workflow.

### Action Fields
| Field | Type | Task 1 | Task 2 | Task 3 |
|---|---|---|---|---|
| category | str | ✓ | ✓ | ✓ |
| priority | str | | ✓ | ✓ |
| department | str | | | ✓ |
| response_action | str | | | ✓ |

Categories: billing, account_access, technical_support, sales, spam, general, compliance, onboarding, feedback
Priorities: critical, high, medium, low
Departments: finance, support, engineering, sales, operations, compliance, hr
Response actions: respond, escalate, forward, ignore, acknowledge

### Observation Fields
- task_id, task_name, instructions, allowed_fields (from task definition)
- current_email (subject, sender, body — NOT ground truth fields)
- inbox_size, emails_remaining, emails_processed
- history (previous actions + scores in this episode)
- Inherited: done, reward, metadata

### State Fields
- Inherited: episode_id, step_count
- current_task_id, seed
- inbox (full email list with IDs)
- current_email_index
- per_email_scores (list of floats)
- total_reward

### Episode Flow
1. `reset(seed, task_id)` → loads inbox of 3–5 random emails, returns first email observation
2. `step(action)` → grades action for current email, advances to next email, returns observation
3. When all emails processed → `observation.done = True`, `observation.reward = trajectory_reward`
4. Per-step reward placed in `observation.reward`

### Three Tasks
| ID | Name | Difficulty | Fields | What Makes It Hard |
|---|---|---|---|---|
| 1 | Category Classification | easy | category only | Straightforward keyword matching |
| 2 | Category + Priority | medium | category, priority | Priority requires context reasoning |
| 3 | Full Triage Routing | hard | all 4 fields | Ambiguous emails, context-dependent priority, thread references |

### Hard Task Challenges for Frontier Models
- Ambiguous emails: billing complaint about a technical issue — could be either category
- Context-dependent priority: same issue is "high" from enterprise client, "medium" from free tier
- Thread awareness: email references a previous email in the inbox
- Response action reasoning: escalate vs respond depends on whether it's a repeat contact

### Grader Design
- Per-field scoring with weights per task (same as existing grader)
- Partial credit via category similarity matrix
- Priority proximity scoring
- Exact match for department and response_action
- All deterministic, reproducible, score in [0.0, 1.0]

### Reward Design
- Per-step: grader score for current email → `observation.reward`
- Trajectory: final observation.reward = weighted avg of all per-email scores
- Step penalty: -0.03 per step beyond inbox_size (if agent takes extra steps somehow)
- Clamped [0.0, 1.0]

---

## 6. DATASET REQUIREMENTS

- 40+ labeled business emails with ground truth for all fields
- Diverse categories covering all 9 categories
- Edge cases: ambiguous, misleading subjects, mixed signals
- Some emails that reference other emails (for thread-awareness in hard task)
- Each record: email_id, subject, sender, body, category, priority, department, response_action
- Optional for hard task: ambiguity_note, related_email_id

---

## 7. IMPLEMENTATION PHASES

### Phase 0: Fresh Project Setup
1. Delete old `env/`, `api/`, `baseline/`, `__pycache__/` dirs and root `Dockerfile`
2. Install openenv: `pip install git+https://github.com/meta-pytorch/OpenEnv.git`
3. Create `server/` directory with `__init__.py`
4. Create `pyproject.toml` for pip-installability

### Phase 1: Core Models + Environment (P0)
1. Create `models.py` at root — Action/Observation/State extending openenv base types
2. Move + adapt `grader.py`, `reward.py`, `tasks.py` into `server/`
3. Create `server/environment.py` — EmailTriageEnvironment extending Environment base
4. Multi-step inbox episodes: reset loads inbox, step processes one email at a time
5. Test: instantiate env, reset, step through inbox, verify done/reward flow

### Phase 2: Server + Client (P0)
1. Create `server/app.py` — use `create_app()` + add custom `/tasks` and `/grader` routes
2. Create `server/Dockerfile` — python:3.11-slim, port 7860
3. Create `client.py` — EnvClient subclass for typed access
4. Test: `uvicorn server.app:app` → hit endpoints → verify /health returns {"status":"healthy"}

### Phase 3: Inference Script (P0 — blocks submission)
1. Create `inference.py` at root
2. Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
3. Connects to local server or HF Space
4. Loops: for each task, for each email → reset → prompt LLM → parse → step → collect score
5. Prints per-task and overall scores
6. Test: run against local server with LLM API

### Phase 4: Dataset Expansion (P0)
1. Expand `data/dataset.json` from 6 to 40+ emails
2. Add ambiguous, edge case, and thread-referencing emails
3. Verify grader produces varied scores across dataset
4. Re-run baseline and record scores

### Phase 5: Validation & Docker (P1)
1. Run `openenv validate` — fix any issues
2. `docker build -t email-triage .` from server/Dockerfile
3. `docker run -p 7860:7860 email-triage`
4. Run inference.py against Docker container
5. Verify 2 vCPU / 8 GB constraint

### Phase 6: Deploy to HF Spaces (P1)
1. Create HF Space (Docker type) tagged `openenv`
2. Push code (git or `openenv push`)
3. Verify: Space returns 200, all endpoints work
4. Run inference.py against live Space URL

### Phase 7: Polish (P2)
1. README: environment description, action/observation schemas, task descriptions, setup, baseline scores
2. Run pre-submission validation script
3. Fresh clone → docker build → all endpoints → inference.py → passes

---

## 8. WHAT TO KEEP FROM OLD SCAFFOLD

### Keep (logic is sound, just needs interface adaptation)
- `env/grader.py` → move to `server/grader.py` (category similarity, priority scoring, per-task weights)
- `env/reward.py` → move to `server/reward.py` (step penalty + clamping)
- `env/tasks.py` → move to `server/tasks.py` (TASKS dict, load_dataset)
- `data/dataset.json` → keep + expand
- `openenv.yaml` → keep + update entry_point

### Discard (doesn't match openenv standards)
- `env/models.py` → replace with root `models.py` extending openenv base types
- `env/environment.py` → replace with `server/environment.py` extending Environment base
- `api/server.py` → replace with `server/app.py` using create_app()
- `baseline/baseline_agent.py` → replace with `inference.py`
- Root `Dockerfile` → replace with `server/Dockerfile`
- `env/__init__.py` → not needed in new structure

### Bug Fixes Needed
- Old `baseline_agent.py` used `client.responses.create()` — wrong API. inference.py uses `client.chat.completions.create()`
- Old baseline read `OPENAI_API_KEY` — must use `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

---

## 9. KEY TECHNICAL DECISIONS

| Decision | Rationale |
|---|---|
| Email triage domain | Highest real-world utility weight (30%), genuine enterprise task |
| Extend openenv base classes | Required for openenv validate, framework integration |
| create_app() for server | Auto-provides HTTP + WebSocket + /health + /docs |
| Multi-step inbox episodes | Meaningful trajectory rewards, tests consistency |
| 40+ email dataset | Enough variance for agentic evaluation, diverse scoring |
| Heuristic + LLM inference | Heuristic for local dev, LLM for submission |
| WebSocket support via framework | Automatic from create_app(), needed for scaling eval |
| pyproject.toml | Makes env pip-installable from HF Space (framework convention) |

---

## 10. RISK REGISTER

| Risk | Impact | Mitigation |
|---|---|---|
| openenv package won't install or has breaking changes | Blocks everything | Test install first, pin version if needed |
| create_app() doesn't support custom endpoints | Server incomplete | Add routes to app object after create_app() |
| Grader always returns same score (DQ) | Fatal | Test grader with varied inputs, verify score variance |
| inference.py exceeds 20 min | Fails validation | ~40 emails × 3 tasks × ~5s/call = ~10 min, well within limit |
| 2 vCPU / 8 GB not enough | Fails validation | Lightweight Python + FastAPI, no ML at runtime |
| Hard task too easy for frontier models | Low task quality score | Add genuine ambiguity, context-dependence, thread-awareness |
| Docker build fails on HF Spaces | Blocks deployment | Test locally first, use standard python:3.11-slim base |

---

## 11. SAMPLE REFERENCE CODE

### How create_app() works (from OpenEnv docs)
```python
from openenv.core.env_server import Environment, create_app
env = MyEnvironment()
app = create_app(env, MyAction, MyObservation)
# app is a FastAPI app with /reset, /step, /state, /health, /ws auto-registered
```

### How inference.py should look (adapted from sample)
```python
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
# Use client.chat.completions.create(model=MODEL_NAME, messages=[...])
```

### How Environment subclass works
```python
from openenv.core.env_server import Environment
class EmailTriageEnvironment(Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]):
    def reset(self, seed=None, episode_id=None, **kwargs):
        # Load inbox, return first email observation
    def step(self, action, timeout_s=None, **kwargs):
        # Grade action, advance inbox, return observation with .done and .reward
    @property
    def state(self):
        # Return current EmailTriageState
```
