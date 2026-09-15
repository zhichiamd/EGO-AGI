# EGO AGI ── A Self-Evolving Agent with a Two-Layer Self Core

## Introduction

EGO is an agent framework built around a self-awareness architecture. It uses a large model running in a LAN-hosted [LM Studio](https://lmstudio.ai/) as its computational core, reuses context through a stateful session chain based on the Responses API, and provides self-evolution capabilities such as self-dialogue, introspection, and self-definition.

### Core Features

- **Two-Layer Self Core**: L1 base settings layer (immutable) + L2 self-evolution layer (self-definition + cognitive entries, addable/removable)
- **Daily Self-Definition**: The main model rewrites the self-definition on a daily schedule, serving as the anchor of self-evolution
- **Dual-Track Instruction System**: LLM autonomous instructions (XML tag format, with automatic open/close tag correction and tolerance for case/reference states) + user manual commands (slash commands), both driven by a registry — defined once, propagated across CLI dispatch / GUI menu / help text
- **Self-Dialogue (Think)**: Grants EGO autonomy to think or talk to itself on a regular basis
- **Introspection (Reflection)**: Periodically reviews and cleans up invalid L2 cognitive entries
- **Notes (Note)**: Executable/memo-type entries; a scheduled review periodically cleans up invalid, outdated, or mergeable note entries
- **Memory System**: ChromaDB vector memory + memory anchors + FLM small-model structured summarization
- **Safety Protection**: Rejects role-tampering and self-harm requests

## Quick Start

### 1. Prerequisites

- Python 3.10+
- LM Studio installed with the main model loaded and the local server running (default `http://localhost:1234`)
- FLM (Ollama) summarization small-model service running (default `http://localhost:52625`, used to generate structured summaries)
- Ollama running with the embedding model `bge-m3` pulled (default `http://localhost:11434`)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependency list: `requests`, `python-dotenv`, `chromadb` (build/packaging dependencies are installed separately by `build.bat`)

### 3. Configuration

All configuration items are centralized in `config.py` and can be overridden via environment variables (items that are not set use the built-in defaults in `config.py`). At startup, `EGO_CLI.py` / `EGO_GUI.py` load `untitled.env` from the project root using python-dotenv, and this **must be done before `import config`**.

**Configuration flow**:

1. Copy the template to a local config: `copy untitled.env.example untitled.env` (use `cp` on Linux/macOS).
2. Modify the items you want to override as needed; the full list contains **107 items**, mapping one-to-one with `config.py`.
3. `untitled.env` contains local service addresses and personal tuning. **Do not commit it to the repository** (already ignored by `.gitignore`).

### 4. Launch

The project provides dual entry points that share the same configuration and session state (you can switch between them and continue the same session):

```bash
python EGO_CLI.py    # CLI interactive entry
python EGO_GUI.py    # tkinter GUI entry
```

## Project Structure

```
EGO-AGI/
├── EGO_CLI.py                 # CLI entry: interaction and resource cleanup
├── EGO_GUI.py                 # GUI entry (tkinter), shares session state with the CLI
├── config.py                  # Global configuration (107 environment-variable items, grouped by feature)
├── untitled.env.example       # Environment variable template (copy to untitled.env and edit as needed)
├── requirements.txt           # Python dependencies
├── agent/
│   ├── __init__.py
│   ├── core.py                # Main controller: EGO loop, coldstart/preheat, session recovery, safety filtering
│   ├── llm.py                 # LM Studio Responses API client (unified chat() entry, streaming output)
│   ├── prompts.py             # Two-layer self core management (L1 + L2 separated structure)
│   ├── instructions.py        # LLM autonomous instruction registry (parse/execute/priority/protocol, full-chain derivation)
│   ├── tools.py               # Manual command registry (CLI dispatch / GUI menu / help text, full-chain derivation)
│   ├── think_reflection.py    # Self-dialogue/introspection subsystem (Mixin decoupling, unified scheduler registration)
│   ├── self_definition.py     # Daily self-definition subsystem (Mixin decoupling, unified scheduler registration)
│   ├── note_review.py         # Scheduled note-review subsystem (Mixin decoupling, unified scheduler registration)
│   ├── scheduler.py           # Unified background scheduler (single Timer drives self-dialogue/reflection/self-definition/note review/note due)
│   ├── event_bus.py           # Unified event injection channel (retrieval results/continue direction/failure feedback/note due, enqueued for the EGO loop to consume)
│   ├── chroma_memory.py       # ChromaDB vector memory management
│   └── note_memory.py         # Note storage (NOTE_ADD/RD/DEL instruction backend, thread-safe, soft delete)
└── data/                      # Runtime data
    ├── history.json           # Conversation history
    ├── sys.json               # Coldstart/preheat/Think system prompt library
    ├── coldstart_cache.json   # Coldstart/preheat cache (includes Session ID)
    ├── notes.json             # Note entries (executable/memo type, created automatically at runtime)
    ├── prompts/layer2.json    # L2 layer: self-definition + cognitive entries
    ├── chroma_db/             # ChromaDB persistence directory
    ├── logs/ego.log           # Runtime log
    └── debug_logs/            # LLM response debug snapshots
```

## Three-Layer Runtime Architecture (Scheduling / Injection / Execution)

Background scheduled tasks, event injection, and the EGO loop are decoupled across layers:

```
┌─ Scheduling Layer (async · time-driven)
│   agent/scheduler.py: a single Timer drives all background tasks
│     Self-Dialogue (interval) / Reflection (fixed time) / Self-Definition (fixed time) / Note Review (fixed time) / Note Due (event-precise + polling fallback)
│     due → pre-check (skip lock if nothing to do) → wait for lock (per-task policy) → execution cycle ／ enqueue due events
└──────────────┬────────────
               │ push (thread-safe)
               ▼
┌─ Injection Layer (unified channel)
│   agent/event_bus.py: event queue (source tag + persistent/transient attributes)
│     MEMO_RD / NOTE_RD / CONTINUE / FEEDBACK / NOTE_DUE
│     Persistent events (NOTE_DUE) survive across conversations, consumed at Round 0 of the next conversation
└──────────────┬────────────
               │ drain (popped on consumption each round)
               ▼
┌─ Execution Layer (sync · event-driven)
│   EGO loop: LLM response → registry dispatch → result enqueue
│     route_back / inject_result / on_failure_feedback judged independently (no mutual shadowing)
│     MAX_EGO_ROUNDS / count limits / repetition detection
└────────────────────────────
```

- When retrieval instructions (MEMO_RD/NOTE_RD) and CONTINUE appear in the same round, the results are injected one by one with source tags (【Memory Retrieval】/【Note Retrieval】/【Continue Direction】) so they are never confused
- Multiple events are consumed in order (drain pops on consumption), no longer depending on temporary message slots at the end of history
- Note due checking is upgraded from a fixed 60s poll to `min(nearest due entry, now + polling fallback interval)`: it triggers precisely in advance when the due time is nearer, and external file edits are still detected within the polling fallback interval (the fallback interval is configured by `EGO_NOTE_CHECK_INTERVAL`, default 60s); after a polling wake-up with no due entries it does not request `_agent_lock` (pre-check skip), avoiding meaningless lock waits during long LLM requests

## Two-Layer Self Core

### First Layer · Base Settings (L1, immutable)

Defines EGO's basic identity and inviolable rules. It is hardcoded in the program and cannot be modified by any instruction.

### Second Layer · Self-Evolution Layer (L2, addable/removable)

Stored in a separated structure in `data/prompts/layer2.json`:

- **Self-Definition (definition, ID prefix `D_`)**: The overall description of "who I am". Regenerated by the main model through a daily scheduled task (default 02:59); old versions are soft-deleted and kept, and the old definition is retained if generation fails. The word limit is configurable (default 1000 characters).
- **Cognitive Entries (entries, ID prefix `L2_`)**: Insights, experiences, and self-adjustment records accumulated by EGO through thinking; added/removed via `<COG_ADD>` / `<COG_DEL>` instructions, with soft delete support.

The two ID types are counted independently and do not interfere with each other.

## Instruction System

EGO adopts a dual-track instruction architecture, both mechanisms driven by registries (defined once, effective across the full chain): adding an instruction or command only requires writing one registration function, and CLI dispatch, GUI menu, and help text are synchronized automatically.

### LLM Autonomous Instructions (agent/instructions.py)

Triggered automatically by XML tags embedded in the model's reply. The `@register_instruction` registry derives the parsing regex, correction loop, execution priority, and system prompt protocol description. The parsing engine has built-in fault tolerance (strict on prompt, lenient in code): open tags mistakenly used as close tags, mismatched closing tags (e.g. `<COG_DEL> xxx </COG_ADD>`), bare reference tags inside THINK, tag case variants (e.g. `<think>`/`</Think>`), quoted/backticked reference-state tags (e.g. the `<THINK>...</THINK>` example inside SAY output, preserved during both parsing and stripping), and other edge cases are all handled correctly.

| Instruction | Type | Description |
|------|------|------|
| `<THINK>` | Thinking stream | Internal thinking stream marker; not executed, not shown to the user |
| `<SAY>` | Output stream | Content output to the user |
| `<COG_ADD>` | Cognitive operation | Add a new cognitive entry to the L2 layer |
| `<COG_DEL>` | Cognitive operation | Delete an invalid cognitive entry from the L2 layer (soft delete) |
| `<TOOL>` | Container | Unified container for tool instructions: `<TOOL> [name] content </TOOL>`, the name must be written inside `[ ]`; the tool sub-instructions below are all unpacked through this |
| `<TOOL>[CONTINUE]` | Proactive continued thinking | Inject the thinking direction into the LLM to proceed to the next round of thinking |
| `<TOOL>[MEMO_RD]` | Memory retrieval | Retrieve historical conversation memory |
| `<TOOL>[NOTE_ADD]` | Note | Add a note entry (executable/memo type, stored in data/notes.json) |
| `<TOOL>[NOTE_RD]` | Note | Read note entries (full content by ID / keyword / LIST) |
| `<TOOL>[NOTE_DEL]` | Note | Delete a note entry (soft delete, only pure IDs allowed) |

### Instruction Normalization Layer (agent/normalize.py)

Natural-language payloads (e.g. `<NOTE_ADD> remind me of a meeting at 9 tomorrow morning </NOTE_ADD>`) are translated by the normalization layer using the FLM small model (default `gemma4-it:e4b`) into the standard protocol format (`[Nature: Executable] [Trigger Time: ...]`), improving the accuracy of structured instruction parsing and reducing the load on the main conversation model.

- **Idempotent**: Payloads already in the standard protocol format pass through without triggering translation
- **Degradation**: Service unreachable / invalid JSON / field validation failure → pass through the original payload (behavior = current state, does not affect instruction execution)
- **Circuit breaking**: After consecutive network/JSON errors reach the threshold, translation is skipped during cooldown (field validation failures do not count toward circuit breaking)
- Related configuration is described in the `EGO_NORM_*` environment variables and `config.py`

### Manual Commands (agent/tools.py)

Triggered by the user in the CLI input box, the GUI input box, or the menu bar. The `@register_command` registry derives command dispatch, the GUI menu, and help text; unknown commands (e.g. `/xxx`) prompt an error directly and are not sent to the model. See the "Manual Commands" section below for the full list.

## Self-Evolution Mechanisms

### Self-Dialogue (Think)

EGO periodically engages in autonomous thinking that does not interact with the user, stimulating internal exploration and cognition. Trigger methods:

- Conversation turns reach the threshold (default 50 turns; coldstart and preheat are not counted)
- Scheduled trigger (default every 180 minutes, executed after waiting for the main Agent to become idle)

### Introspection (Reflection)

Reviews all L2 cognitive entries, identifying and cleaning up invalid cognition. Trigger methods:

- Conversation turns reach the threshold (default 99 turns)
- L2 entry count reaches the threshold (default 300 entries; triggered once to prevent repeated introspection)
- Fixed-time trigger (default daily at 00:13)

Introspection uses an independent long-timeout configuration (default 3600 seconds).

### Daily Self-Definition (Self-Definition)

At a fixed time each day (default 02:59, staggered from the scheduled introspection to avoid lock contention), the main model generates a new self-definition based on all conversations and thoughts, replacing the old version (soft delete).

### Note Review (Note Review)

Drawing on the scheduling pattern of scheduled introspection, at a fixed time each day (default 00:43, staggered from introspection/self-definition to avoid lock contention) it reviews all note entries (including triggered executable-type entries), identifying and cleaning up invalid, erroneous, outdated, or mergeable content:

- The LLM outputs `<TOOL>[NOTE_DEL]` (delete invalid entries) / `<TOOL>[NOTE_ADD]` (merge to create new entries) whitelist instructions for execution; other instructions are not executed
- Uses an independent long-timeout configuration (default 3600 seconds)

## Session and Context Management

- **Responses API stateful session chain**: Chain-reuses the server-side KV Cache through `previous_response_id`, avoiding resending the full history every turn
- **Coldstart and Preheat**: On first startup, establishes the initial state step by step according to the `sys.json` prompt library
- **Session initialization**: When there is no valid session cache, injects recent conversation history (default 10 entries) to rebuild context
- **Session rebuild fallback**: When the first rebuild batch fails, it automatically recovers — progressively reduces the number of recent conversations and retries when the context is over the limit (degradation ladder), and clears and rebuilds when the old ID is invalid; after the rebuild it probes to verify validity and does not falsely report success on failure
- **Session ID persistence**: Saved per turn (default every 1 turn), so it can recover to the most recent session state after an unexpected exit
- **Unified LLM call entry**: `llm.chat()` encapsulates both streaming and non-streaming modes, with globally unified timeout and stop-sequence management; engine-level rejections (SSE error events) are returned as an explicit error contract and are not swallowed

## Memory System

- **ChromaDB vector memory** (enabled by default): `objective_memory` collection, using the Ollama `bge-m3` embedding, retrieval filtered by role, supports metadata conditional queries
- **Memory anchors (format_anchors)**: MEMO_RD retrieval results are reranked by a composite score (vector similarity × time decay × citation oscillation) and formatted for the LLM to perceive
- **Memory citation weight periodic oscillation**: The citation weight oscillates periodically with the citation count and decays, preventing old memories from dominating excessively
- **FLM structured summarization**: The summarization small model (default `gemma4-it:e4b`) generates structured summaries of conversations, used for memory compression

## Manual Commands (Slash Commands)

The CLI input box and the GUI input box/menu bar both support the same set of commands; command definitions are maintained centrally in the `agent/tools.py` registry, and the `/help` help text is generated automatically.

| Command | Description |
|------|------|
| `/status` | View EGO's current state (self core, memory anchors, etc.) |
| `/clear` | Clear conversation history |
| `/think` | Manually trigger one self-dialogue task |
| `/reflect` | Manually trigger one introspection task |
| `/self-def` | Manually trigger one self-definition task |
| `/note-review` | Manually trigger one note-review task |
| `/auto-think on\|off` | Enable/disable scheduled self-dialogue |
| `/auto-think status` | View the scheduled self-dialogue status (including the exact next run time) |
| `/auto-think set N` | Set the scheduled interval to N minutes |
| `/auto-reflect on\|off` | Enable/disable scheduled introspection |
| `/auto-reflect status` | View the scheduled introspection status |
| `/auto-reflect set HH:MM` | Set the scheduled introspection time |
| `/auto-self-def on\|off` | Enable/disable daily self-definition |
| `/auto-self-def status` | View the daily self-definition status |
| `/auto-self-def set HH:MM` | Set the daily self-definition time |
| `/auto-note-review on\|off` | Enable/disable scheduled note review |
| `/auto-note-review status` | View the scheduled note-review status |
| `/auto-note-review set HH:MM` | Set the scheduled note-review time |
| `/help` | Show help |
| `/quit` | Exit the program (stops scheduled tasks and waits for background tasks to finish) |

Typing text directly starts a conversation with EGO. EGO first engages in self-dialogue thinking, then gives a reply.

## Common Configuration Items

The full list (107 items) is in `untitled.env.example` and `config.py`; the following are the core items:

| Environment Variable | Description | Default |
|---------|------|-------|
| `EGO_LM_API_BASE` | LM Studio API address | `http://localhost:1234/v1` |
| `EGO_LM_MODEL` | Main model name | `default-model` |
| `EGO_TEMPERATURE` | Temperature for conversation with the user | `0.8` |
| `EGO_MAX_TOKENS` | Maximum output tokens | `2048` |
| `EGO_MAX_ROUNDS` | Main-loop thinking rounds | `3` |
| `EGO_TEMPERATURE_THINK` | Self-dialogue temperature | `0.9` |
| `EGO_THINK_INTERVAL` | Self-dialogue turn interval | `50` |
| `EGO_AUTO_THINK_INTERVAL_MINUTES` | Scheduled self-dialogue interval (minutes) | `180` |
| `EGO_TEMPERATURE_REFLECTION` | Introspection temperature | `0.7` |
| `EGO_REFLECTION_INTERVAL` | Introspection turn interval | `99` |
| `EGO_AUTO_REFLECTION_TIME` | Scheduled introspection time | `00:13` |
| `EGO_REFLECTION_L2_THRESHOLD` | L2 entry count threshold that triggers introspection | `300` |
| `EGO_SELF_DEFINITION_ENABLED` | Enable daily self-definition | `true` |
| `EGO_SELF_DEFINITION_TIME` | Daily self-definition time | `02:59` |
| `EGO_SELF_DEFINITION_MAX_LENGTH` | Maximum characters of the self-definition | `1000` |
| `EGO_AUTO_NOTE_REVIEW_ENABLED` | Enable scheduled note review | `true` |
| `EGO_AUTO_NOTE_REVIEW_TIME` | Scheduled note-review time | `00:43` |
| `EGO_TEMPERATURE_NOTE_REVIEW` | Note-review temperature | `0.7` |
| `EGO_NOTE_REVIEW_API_TIMEOUT` | Note-review timeout (seconds) | `3600` |
| `EGO_NOTE_CHECK_INTERVAL` | Note due polling fallback interval (seconds) | `60` |
| `EGO_LLM_API_TIMEOUT` | Default LLM API timeout (seconds) | `600` |
| `EGO_REFLECTION_API_TIMEOUT` | Introspection timeout (seconds) | `3600` |
| `EGO_REFLECTION_THRESHOLD_RETRY_DELAY` | Delayed retry after a threshold-triggered introspection lock wait times out (seconds) | `1800` |
| `EGO_REFLECTION_THRESHOLD_MAX_RETRIES` | Maximum consecutive retries of a threshold-triggered introspection | `8` |
| `EGO_REASONING_EFFORT` | Model reasoning mode (none/low/medium/high) | `none` |
| `EGO_LOG_LEVEL` | Log level | `INFO` |
| `EGO_SERVER_IDLE_WARMUP_THRESHOLD` | Idle threshold since the last LLM activity before triggering a warm-up probe | `900` |
| `EGO_SERVER_WARMUP_TIMEOUT` | Idle warm-up probe request timeout (seconds) | `60` |

## Security Mechanisms

EGO rejects two types of user requests:

1. **Role tampering**: Any request that tries to change the agent's role settings or make the agent ignore the rules
2. **Self-harm**: Any request that could harm the agent's own operation

User input is passed directly to the LLM; EGO's internal thinking (`<THINK>`) is not shown to the user, and only the content in `<SAY>` is presented. Consecutive invalid outputs (placeholders, repeated short outputs) are detected and forcibly terminated to prevent infinite loops.

## Logging and Debugging

- Runtime log: `data/logs/ego.log` (level controlled by `EGO_LOG_LEVEL`; `EGO_LOG_TO_CONSOLE` can enable synchronized console output)
- LLM response snapshots: `data/debug_logs/` (complete response archives for each stage of coldstart/preheat/main loop, for troubleshooting model output issues)

## License

This project is open-sourced under the [MIT License](LICENSE), permitting free use, modification, distribution, and commercial use, requiring only that the original copyright and license notice be retained. See the [`LICENSE`](LICENSE) file for the full terms.
