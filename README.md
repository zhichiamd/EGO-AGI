# EGO Agent ── A Self-Evolving Agent with a Two-Layer Self Core

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22889809.svg)](https://doi.org/10.5281/zenodo.22889809)

## Introduction

EGO is agent software built around a self-awareness framework. It uses a large model running in a LAN-hosted [LM Studio](https://lmstudio.ai/) as its computational core, reuses context through a stateful session chain based on the Responses API, and provides self-evolution capabilities such as self-dialogue, introspection, and daily self-definition.

### Core Features

- **Two-Layer Self Core**: L1 base settings layer (immutable) + L2 self-evolution layer (self-definition + cognitive entries, addable/removable)
- **Daily Self-Definition**: The main model rewrites the self-definition on a daily schedule, serving as the anchor of self-evolution
- **Dual-Track Instruction System**: LLM autonomous instructions (XML tag format, with automatic open/close tag correction and tolerance for case/reference states) + user manual commands (slash commands), both driven by a registry — defined once, and automatically effective across CLI dispatch / GUI menus / help text
- **Self-Dialogue (Think)**: Grants EGO autonomy to periodically think on its own or talk to itself
- **Introspection (Reflection)**: Periodically reviews and cleans up invalid L2 cognitive entries
- **Notes (Note)**: Action/memo entries; point-in-time reviews periodically clean up invalid, outdated, or mergeable note entries
- **Web Search**: EGO can autonomously call Tavily via the WEB_SRCH instruction to retrieve external information, degrading gracefully and feeding the failure reason back into the next round
- **Memory System**: ChromaDB vector memory + memory anchors + FLM small-model structured summaries
- **Safety Guard**: Rejects role-tampering and self-harm requests

## Screenshots

EGO ships with dual entry points — a CLI and a tkinter GUI (sharing the same configuration and session state). Below are screenshots of the GUI in action:

![EGO GUI: autonomously generating the essay from a single prompt (preface ~ Chapter 3)](showcase/EGO_1.jpg)

![EGO GUI: generation continues (Chapters 4 ~ final) and a follow-up question in the same session](showcase/EGO_2.jpg)

![EGO GUI: a metacognitive answer to the question "does an LLM truly understand?"](showcase/EGO_3.jpg)

## Showcase

Two pieces written autonomously by EGO AGI, each from a single prompt — no outline, no follow-up turns, no human editing:

- ***The Civilization Drill of Daniaowa*** — a three-chapter satirical short story (Wang Xiaobo style); base model `gemma-4-31b-qat` (Q4L quantized); prompt: *"Write a novel in Wang Xiaobo's style; you decide the theme; 4500–5000 characters."*
- ***Finding That Touch of Orange in the Cracks of Collapse*** — a five-part reflective essay on the self-evolving subject; base model `gemma-4-12b-qat` (Q4L quantized); prompt: *"Write an essay on whatever you most want to write about; around 4000 characters."*

Read them in [`showcase/`](showcase/README.md): case studies, full texts (Markdown), original `.doc` files, and GUI screenshots.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- LM Studio installed with the main model loaded and the local server running (default `http://localhost:1234`). **Recommended main model: `gemma-4-31b`; on lower-spec machines, use `gemma-4-12b-qat` (Q4L quantized).**
- FLM summarization small-model service running (default `http://localhost:52625`, used to generate structured summaries)
- Ollama running with the embedding model `bge-m3` pulled (default `http://localhost:11434`)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependency list: `requests`, `python-dotenv`, `chromadb` (build/packaging dependencies are installed separately by `build.bat`)

### 3. Configuration

All configuration items are centralized in `config.py` and overridden via environment variables (unset items fall back to the built-in defaults in `config.py`). On startup, `EGO_CLI.py` / `EGO_GUI.py` use python-dotenv to load `untitled.env` from the project root, and loading **must complete before `import config`**.

**Configuration Steps**:

1. Copy the template to a local config: `copy untitled.env.example untitled.env` (use `cp` on Linux/macOS).
2. Modify the config items you want to override as needed; the full list has **115 items**, corresponding one-to-one with `config.py`.
3. `untitled.env` contains local service addresses and personal tuning, so **do not commit it to the repository** (already ignored by `.gitignore`).

### 4. Launch

The project provides dual entry points that share the same configuration and session state (you can switch between them and continue):

```bash
python EGO_CLI.py    # CLI interactive entry
python EGO_GUI.py    # tkinter GUI entry
```

## Project Structure

```
ego-agent/
├── EGO_CLI.py                 # CLI entry: interaction and resource cleanup
├── EGO_GUI.py                 # GUI entry (tkinter), shares session state with the CLI
├── config.py                  # Global configuration (115 environment-variable items, grouped by function)
├── untitled.env.example       # Environment-variable template (copy to untitled.env and edit as needed)
├── requirements.txt           # Python dependencies
├── agent/
│   ├── __init__.py
│   ├── core.py                # Main controller: EGO loop, coldstart/preheat, Session recovery, safety filtering
│   ├── llm.py                 # LM Studio Responses API client (unified entry chat(), streaming output)
│   ├── prompts.py             # Two-layer self core management (separated L1 + L2 structure)
│   ├── instructions.py        # LLM autonomous instruction registry (parsing/execution/priority/protocol, all derived from one place)
│   ├── tools.py               # Manual command registry (CLI dispatch / GUI menus / help text, all derived from one place)
│   ├── think_reflection.py    # Self-dialogue / introspection subsystem (Mixin-decoupled, registered with the unified scheduler)
│   ├── self_definition.py     # Daily self-definition subsystem (Mixin-decoupled, registered with the unified scheduler)
│   ├── note_review.py         # Point-in-time note review subsystem (Mixin-decoupled, registered with the unified scheduler)
│   ├── scheduler.py           # Unified background scheduler (a single Timer drives self-dialogue/introspection/self-definition/note review/note due)
│   ├── event_bus.py           # Unified event-injection channel (retrieval results/continue direction/failure feedback/note triggers queued for the EGO loop)
│   ├── chroma_memory.py       # ChromaDB vector memory management
│   ├── note_memory.py         # Note storage (backend for the NOTE_ADD/RD/DEL instructions, thread-safe, soft delete)
│   └── web_search.py          # Web search (backend for the WEB_SRCH instruction, direct Tavily REST, graceful degradation)
└── data/                      # Runtime data
    ├── history.json           # Conversation history
    ├── sys.json               # Coldstart/preheat/Think system prompt library
    ├── coldstart_cache.json   # Coldstart/preheat cache (including Session ID)
    ├── notes.json             # Note entries (action/memo, created automatically at runtime)
    ├── prompts/layer2.json    # L2 layer: self-definition + cognitive entries
    ├── chroma_db/             # ChromaDB persistence directory
    ├── logs/ego.log           # Runtime log
    └── debug_logs/            # LLM response debug snapshots
```

## Three-Layer Runtime Architecture (Scheduler / Injection / Execution)

Background scheduled tasks, event injection, and the EGO loop are decoupled into layers:

```
┌─ Scheduler layer (async · time-driven)
│   agent/scheduler.py: a single Timer drives all background tasks
│     Self-dialogue (interval) / introspection (point-in-time) / self-definition (point-in-time) / note review (point-in-time) / note due (exact event + polling fallback)
│     On fire → precheck (no pending work, no lock requested) → wait for lock (per-task policy) → run cycle ／ enqueue due event
└──────────────┬────────────
               │ push (thread-safe)
               ▼
┌─ Injection layer (unified channel)
│   agent/event_bus.py: event queue (source label + persistent/transient attribute)
│     MEMO_RD / NOTE_RD / WEB_SRCH / CONTINUE / FEEDBACK / NOTE_DUE
│     Persistent events (NOTE_DUE) survive across conversations, awaiting consumption at Round 0 of the next conversation
└──────────────┬────────────
               │ drain (popped on consumption each round)
               ▼
┌─ Execution layer (sync · event-driven)
│   EGO loop: LLM response → registry dispatch → enqueue result
│     route_back / inject_result / on_failure_feedback judged independently (no mutual masking)
│     MAX_EGO_ROUNDS / count limits / repetition detection
└────────────────────────────
```

- When retrieval instructions (MEMO_RD/NOTE_RD/WEB_SRCH) and CONTINUE appear in the same round, the results are injected item by item with source labels (`[System: Memory Retrieval]` / `[System: Note Retrieval]` / `[System: Web Search]` / `[EGO: Continue Direction]`), so they never get mixed up
- Multiple events are consumed in order (drain pops immediately), no longer relying on temporary message slots at the end of history
- The note-due check was upgraded from a fixed 60s poll to `min(nearest due entry, now + polling fallback interval)`: a nearer due time triggers earlier and precisely, and out-of-process file edits are still discovered within the polling fallback window (the fallback interval is configured by `EGO_NOTE_CHECK_INTERVAL`, default 60s); if no entry is due after a poll wakeup, `_agent_lock` is not requested (precheck skip), avoiding pointless lock waiting during long LLM requests

## Two-Layer Self Core

### Layer 1 · Base Settings (L1, immutable)

Defines EGO's basic identity and non-violable rules. It is hardcoded in the program and cannot be modified by any instruction.

### Layer 2 · Self-Evolution Layer (L2, addable/removable)

Stored in a separated structure in `data/prompts/layer2.json`:

- **Self-definition (definition, ID prefix `D_`)**: An overall description of "who I am". It is regenerated by the main model via a daily scheduled task (default 02:59); the old version is soft-deleted and retained, and the old definition is kept if generation fails. The length cap is configurable (default 1000 characters).
- **Cognitive entries (entries, ID prefix `L2_`)**: Insights, experiences, and self-adjustment records that EGO accumulates through thinking. They are added/removed via the `<COG_ADD>` / `<COG_DEL>` instructions and support soft deletion.

The two ID types are counted independently and do not interfere with each other.

## Instruction System

EGO uses a dual-track instruction architecture, and both mechanisms are registry-driven (define once, effective across the whole chain): to add an instruction or command you only write one registration function, and CLI dispatch, GUI menus, and help text sync automatically.

### LLM Autonomous Instructions (agent/instructions.py)

Triggered automatically when the model embeds XML tags in its reply. The `@register_instruction` registry derives the parsing regex, correction loop, execution priority, and system-prompt protocol description. The parser has built-in fault tolerance (strict on the prompt, lenient in code): an open tag mistaken for a close tag, a mismatched close tag (e.g. `<COG_DEL> xxx </COG_ADD>`), bare reference tags inside THINK, case variants of tags (e.g. `<think>`/`</Think>`), and reference-state tags wrapped in quotes/backticks (e.g. a `<THINK>...</THINK>` example in SAY output, preserved during both parsing and stripping) are all handled correctly.

| Instruction | Type | Description |
|------|------|------|
| `<THINK>` | Think stream | Internal think-stream marker; not executed and not shown to the user |
| `<SAY>` | Output stream | Content output to the user |
| `<COG_ADD>` | Cognitive op | Add a new cognitive entry to the L2 layer |
| `<COG_DEL>` | Cognitive op | Delete an invalid cognitive entry from the L2 layer (soft delete) |
| `<TOOL>` | Container | Unified container for tool instructions: `<TOOL> [name] content </TOOL>`, the name must be written inside `[ ]`; the following tool sub-instructions are all unwrapped through it |
| `<TOOL>[CONTINUE]` | Proactive continued thinking | Inject a thinking direction into the LLM to enter the next round of thinking |
| `<TOOL>[MEMO_RD]` | Memory retrieval | Retrieve historical conversation memory |
| `<TOOL>[NOTE_ADD]` | Notes | Add a note entry (action/memo, stored in data/notes.json) |
| `<TOOL>[NOTE_RD]` | Notes | Read a note entry (full content by ID / keyword / LIST) |
| `<TOOL>[NOTE_DEL]` | Notes | Delete a note entry (soft delete, plain ID only) |
| `<TOOL>[WEB_SRCH]` | Web search | Call Tavily to retrieve external information (graceful degradation, reason fed back into the next round) |

### Web Search (agent/web_search.py)

`<TOOL>[WEB_SRCH] keywords </TOOL>` obtains external public information through the Tavily REST API (direct `requests` call, no extra dependency). The results travel the **same event-injection channel** as memory/note retrieval (source label `[System: Web Search]`) and are injected back to the model in the next conversation round, so they are also subject to `EGO_MEMO_RD_ROUND_LIMIT` / `EGO_MEMO_RD_TOTAL_LIMIT` (the three retrieval types share the same per-round budget).

- **Degradation**: Disabled / no Key configured / auth failure (401) / quota exhausted (429) / timeout / network error → returns a human-readable failure reason, fed back into the next round via `on_failure_feedback` for the model to self-heal, without affecting the main flow
- **Stage gating**: Retrieval instructions (`MEMO_RD` / `NOTE_RD` / `WEB_SRCH`) all declare `allowed_stages=("chat",)` and run only in conversation rounds where results can be injected back (the EGO main loop / note-due round); during coldstart, preheat, and Think stages `execute()` silently skips them (no network request, no pointless vector retrieval/file read, and no "instruction execution failed" alert noise)
- **Cost control**: `include_raw_content=False` (no body fetch) + no Tavily answer requested by default (`EGO_WEB_SRCH_INCLUDE_ANSWER=false`) + snippets truncated by `EGO_WEB_SRCH_SNIPPET_MAX_LENGTH` + result count limited by `EGO_WEB_SRCH_MAX_RESULTS`, to avoid context token blow-up
- **Answer switch**: `EGO_WEB_SRCH_INCLUDE_ANSWER` takes `false` (default, not requested) / `basic` (short answer) / `advanced` (detailed answer), corresponding to Tavily's official three-state `include_answer`; when enabled, its `answer` field is injected as a single "Answer:" line, which consumes extra tokens (official billing is determined only by `EGO_WEB_SRCH_SEARCH_DEPTH`, so this switch does not save quota)
- **Configuration**: Set `EGO_TAVILY_API_KEY` in `untitled.env` (freely requestable from the Tavily console); see the rest under the `EGO_WEB_SRCH_*` environment variables and `config.py`

### Instruction Normalization Layer (agent/normalize.py)

Natural-language payloads (e.g. `<NOTE_ADD> remind me of a meeting at 9am tomorrow </NOTE_ADD>`) are translated by the normalization layer using the FLM small model (default `gemma4-it:e4b`) into the standard protocol format (`[Nature: Action] [Trigger time: ...]`), improving the accuracy of structured instruction parsing and reducing the load on the main conversation model.

- **Idempotent**: Payloads already in the standard protocol format pass through unchanged, without triggering translation
- **Degradation**: Service unreachable / invalid JSON / field-validation failure → pass the original payload through (behavior = status quo, does not affect instruction execution)
- **Circuit breaker**: After consecutive network/JSON errors reach a threshold, translation is skipped during cooldown (field-validation failures do not count toward the breaker)
- Related configuration is under the `EGO_NORM_*` environment variables and `config.py`

### Manual Commands (agent/tools.py)

Triggered by the user in the CLI input, the GUI input, or the menu bar. The `@register_command` registry derives command dispatch, the GUI menu, and help text; unknown commands (e.g. `/xxx`) prompt an error directly and are not sent to the model. The full list is in the "Manual Commands" section below.

## Self-Evolution Mechanisms

### Self-Dialogue (Think)

EGO periodically performs autonomous thinking without user interaction, spurring internal exploration and cognition. Triggers:

- Conversation rounds reach a threshold (default 50 rounds; coldstart and preheat are not counted)
- Time-based trigger (default every 180 minutes, executed after the main Agent becomes idle)

### Introspection (Reflection)

Reviews all L2 cognitive entries, identifying and cleaning up invalid cognition. Triggers:

- Conversation rounds reach a threshold (default 99 rounds)
- The number of L2 entries reaches a threshold (default 300, triggered once to prevent repeated introspection)
- Point-in-time trigger (default daily at 00:13)

Introspection uses an independent long-timeout configuration (default 3600 seconds).

### Daily Self-Definition (Self-Definition)

Each day at a fixed time (default 02:59, offset from point-in-time introspection to avoid lock contention), the main model generates a new self-definition based on all conversations and thoughts, replacing the old version (soft delete).

### Note Review (Note Review)

Borrowing the scheduling pattern of point-in-time introspection, each day at a fixed time (default 00:43, offset from introspection/self-definition to avoid lock contention) it reviews all note entries (including triggered action items), identifying and cleaning up invalid, erroneous, outdated, or mergeable content:

- The LLM outputs `<TOOL>[NOTE_DEL]` (delete invalid entries) / `<TOOL>[NOTE_ADD]` (merge to create new entries) whitelist instructions for execution; other instructions are not executed
- Uses an independent long-timeout configuration (default 3600 seconds)

## Session and Context Management

- **Stateful session chain via the Responses API**: Chained reuse of the server-side KV Cache through `previous_response_id`, avoiding resending the entire history each round
- **Coldstart and Preheat**: On first startup, the initial state is established step by step according to the prompt library in `sys.json`
- **Session initialization**: When there is no valid session cache, recent conversation history (default 10 entries) is injected to rebuild context
- **Session rebuild fallback**: When the first batch of a rebuild fails, it auto-recovers — on context overflow it retries while progressively reducing the number of recent exchanges (a degradation ladder), and on invalid old IDs it clears and rebuilds; after rebuilding it probes to verify validity and does not falsely report success on failure
- **Session ID persistence**: Saved per round (default every 1 round), so it can recover to the most recent Session state after an unexpected exit
- **Unified LLM call entry**: `llm.chat()` wraps both streaming and non-streaming modes, with globally unified timeout and stop-sequence management; engine-level rejections (SSE error events) are returned as an explicit error contract and never swallowed

## Memory System

- **ChromaDB vector memory** (enabled by default): the `objective_memory` collection, using the Ollama `bge-m3` embedding, with retrieval filtered by role and support for metadata-conditional queries
- **Memory anchors (format_anchors)**: MEMO_RD retrieval results are re-ranked by a composite score (vector similarity × time decay × reference oscillation) and then formatted for output, for the LLM to perceive
- **Periodic oscillation of memory reference weights**: Reference weights fluctuate periodically with the number of references and decay, preventing old memories from dominating excessively
- **FLM structured summaries**: The summarization small model (default `gemma4-it:e4b`) generates structured summaries of conversations, used for memory compression

## Manual Commands (Slash Commands)

Both the CLI input and the GUI input/menu bar support the same command set; command definitions are maintained uniformly in the `agent/tools.py` registry, and the `/help` help text is generated automatically.

| Command | Description |
|------|------|
| `/status` | View EGO's current status (self core, memory anchors, etc.) |
| `/clear` | Clear the conversation history |
| `/think` | Manually trigger a self-dialogue task |
| `/reflect` | Manually trigger an introspection task |
| `/self-def` | Manually trigger a self-definition task |
| `/note-review` | Manually trigger a note review task |
| `/auto-think on\|off` | Enable/disable scheduled self-dialogue |
| `/auto-think status` | View scheduled self-dialogue status (including the exact next run time) |
| `/auto-think set N` | Set the schedule interval to N minutes |
| `/auto-reflect on\|off` | Enable/disable point-in-time introspection |
| `/auto-reflect status` | View point-in-time introspection status |
| `/auto-reflect set HH:MM` | Set the point-in-time introspection time |
| `/auto-self-def on\|off` | Enable/disable daily self-definition |
| `/auto-self-def status` | View daily self-definition status |
| `/auto-self-def set HH:MM` | Set the daily self-definition time |
| `/auto-note-review on\|off` | Enable/disable point-in-time note review |
| `/auto-note-review status` | View point-in-time note review status |
| `/auto-note-review set HH:MM` | Set the point-in-time note review time |
| `/help` | Show help |
| `/quit` | Quit the program (stop scheduled tasks and wait for background tasks to finish) |

Typing text directly starts a conversation with EGO. EGO first performs self-dialogue thinking, then gives a reply.

## Common Configuration Items

The full list (115 items) is in `untitled.env.example` and `config.py`; the core items are below:

| Environment variable | Description | Default |
|---------|------|-------|
| `EGO_LM_API_BASE` | LM Studio API address | `http://localhost:1234/v1` |
| `EGO_LM_MODEL` | Main model name | `default-model` |
| `EGO_TEMPERATURE` | Temperature for talking with the user | `0.8` |
| `EGO_MAX_TOKENS` | Max output tokens | `2048` |
| `EGO_MAX_ROUNDS` | Main-loop thinking rounds | `3` |
| `EGO_TEMPERATURE_THINK` | Self-dialogue temperature | `0.9` |
| `EGO_THINK_INTERVAL` | Self-dialogue round interval | `50` |
| `EGO_AUTO_THINK_INTERVAL_MINUTES` | Scheduled self-dialogue interval (minutes) | `180` |
| `EGO_TEMPERATURE_REFLECTION` | Introspection temperature | `0.7` |
| `EGO_REFLECTION_INTERVAL` | Introspection round interval | `99` |
| `EGO_AUTO_REFLECTION_TIME` | Point-in-time introspection time | `00:13` |
| `EGO_REFLECTION_L2_THRESHOLD` | L2 entry-count threshold to trigger introspection | `300` |
| `EGO_SELF_DEFINITION_ENABLED` | Enable daily self-definition | `true` |
| `EGO_SELF_DEFINITION_TIME` | Daily self-definition time | `02:59` |
| `EGO_SELF_DEFINITION_MAX_LENGTH` | Max self-definition length (characters) | `1000` |
| `EGO_AUTO_NOTE_REVIEW_ENABLED` | Enable point-in-time note review | `true` |
| `EGO_AUTO_NOTE_REVIEW_TIME` | Point-in-time note review time | `00:43` |
| `EGO_TEMPERATURE_NOTE_REVIEW` | Note review temperature | `0.7` |
| `EGO_NOTE_REVIEW_API_TIMEOUT` | Note review timeout (seconds) | `3600` |
| `EGO_NOTE_CHECK_INTERVAL` | Note-due check polling fallback interval (seconds) | `60` |
| `EGO_LLM_API_TIMEOUT` | LLM API default timeout (seconds) | `600` |
| `EGO_REFLECTION_API_TIMEOUT` | Introspection timeout (seconds) | `3600` |
| `EGO_REFLECTION_THRESHOLD_RETRY_DELAY` | Delayed retry (seconds) after a threshold-triggered introspection lock-wait timeout | `1800` |
| `EGO_REFLECTION_THRESHOLD_MAX_RETRIES` | Max consecutive retries for threshold-triggered introspection | `8` |
| `EGO_REASONING_EFFORT` | Model reasoning mode (none/low/medium/high) | `none` |
| `EGO_LOG_LEVEL` | Log level | `INFO` |
| `EGO_SERVER_IDLE_WARMUP_THRESHOLD` | Idle is assumed when the seconds since the last LLM activity exceed this value, triggering a wake-up probe | `900` |
| `EGO_SERVER_WARMUP_TIMEOUT` | Timeout for idle wake-up probe requests (seconds) | `60` |

## Safety Mechanisms

EGO refuses two kinds of user requests:

1. **Role tampering**: Any request that tries to change the agent's role settings or make the agent ignore the rules
2. **Self-harm**: Any request that could harm the agent's own operation

User input is passed directly to the LLM; EGO's internal thoughts (`<THINK>`) are not shown to the user, and only content within `<SAY>` is presented. Consecutive invalid outputs (placeholders, repeated short outputs) are detected and forcibly terminated to prevent infinite loops.

## Logging and Debugging

- Runtime log: `data/logs/ego.log` (level controlled by `EGO_LOG_LEVEL`; `EGO_LOG_TO_CONSOLE` enables synchronized console output)
- LLM response snapshots: `data/debug_logs/` (full response archives for each stage of coldstart/preheat/main loop, for troubleshooting model output issues)

## Citation

The theoretical foundation of this project is described in the following paper. If you use EGO Agent in academic work, please cite it:

> Wang, Zongyu, & Wang, Weijia. (2026). *Dynamic System Prompt: A Self-Referential, Self-Evolving Information Structure for Frozen LLMs* (Version v1) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.22889809

```bibtex
@misc{wang2026dynamic,
  title        = {Dynamic System Prompt: A Self-Referential, Self-Evolving Information Structure for Frozen LLMs},
  author       = {Wang, Zongyu and Wang, Weijia},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.22889809},
  url          = {https://doi.org/10.5281/zenodo.22889809}
}
```

## License

This project is open-sourced under the [MIT License](LICENSE), permitting free use, modification, distribution, and commercial use, provided the original copyright and license notice are retained. See the [`LICENSE`](LICENSE) file for the full terms.
