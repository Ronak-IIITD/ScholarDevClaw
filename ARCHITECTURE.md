# ScholarDevClaw Architecture

## The Pitch

**What if your AI assistant could not just chat with you, but actually read your codebase, understand your research papers, find optimization opportunities, and implement them — all while you maintain full control?**

That's ScholarDevClaw.

We're building a **research-to-engineering operating system** that bridges the gap between cutting-edge AI research and production code. While other tools give you generic code suggestions, ScholarDevClaw specifically targets:

1. **Researchers** who want to operationalize their AI/ML innovations
2. **Developers** who need to integrate research into real codebases safely
3. **Teams** who want automated code improvement with human oversight

Think of it as having a seniorStaff Engineer who:
- Reads the latest arXiv papers before you do
- Understands your entire codebase context
- Generates precise patches backed by validation
- Explains *why* each change matters

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ScholarDevClaw                                │
│                 "Research-to-Code Operating System"                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │  CLI/TUI     │   │  FastAPI     │   │  TypeScript  │           │
│  │  Interface   │   │  Server      │   │  Orchestrator│           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            ▼                                        │
│              ┌─────────────────────────────┐                       │
│              │   Application Pipeline      │                       │
│              │   (Shared Workflow Layer)   │                       │
│              └──────────────┬──────────────┘                       │
│                             │                                       │
│    ┌────────────┬───────────┼───────────┬────────────┐           │
│    ▼            ▼           ▼           ▼            ▼             │
│ ┌──────┐  ┌──────────┐ ┌───────┐ ┌─────────┐ ┌──────────┐       │
│ │ Repo │  │ Research │ │Mapping│ │  Patch  │ │Validation│       │
│ │Intel │  │ Intel    │ │       │ │Generat. │ │          │       │
│ └──────┘  └──────────┘ └───────┘ └─────────┘ └──────────┘       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │                    Core Python Engine                    │        │
│  │  Auth | Rollback | Security | Context | Plugins | etc.  │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Leverages
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          OpenClaw                                    │
│                   "Personal AI Assistant Platform"                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                      Gateway                                │     │
│  │         (Control Plane - TypeScript/Node.js)               │     │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐   │     │
│  │  │ Heartbeat│  │ Phase    │  │Checkpoint│  │ Guard   │   │     │
│  │  │ Monitor │  │Orchestrat│  │ Recovery │  │Rails    │   │     │
│  │  └─────────┘  └──────────┘  └─────────┘  └─────────┘   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                 │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐          │
│  │  Telegram  │      │  Discord   │      │   Slack    │          │
│  │  Channel   │      │  Channel   │      │  Channel   │          │
│  └────────────┘      └────────────┘      └────────────┘          │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
│                    ┌─────────────────┐                              │
│                    │   Claude/       │                              │
│                    │   OpenAI Models │                              │
│                    └─────────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How OpenClaw Works (The Parent Engine)

### The Core Philosophy

OpenClaw is a **personal AI assistant platform** that runs on your own devices. It's not a SaaS — it's software you own and control.

### Architecture Layers

#### 1. Gateway (Control Plane)
The Gateway is the heart of OpenClaw — a TypeScript/Node.js service that:
- **Manages lifecycle**: Starts, stops, resumes agent runs
- **Heartbeat monitoring**: Keeps track of running processes, recovers from crashes
- **Phase orchestration**: Breaks down complex tasks into sequential phases
- **Checkpoint persistence**: Saves state after each phase for resume capability
- **Guardrails**: Enforces safety policies before dangerous operations

```
Gateway Flow:
User Message → Router → Agent Engine → [Phase 1] → [Phase 2] → ... → Response
                    │
                    ▼
            ┌───────────────┐
            │  Phase Result │
            │   Persisted   │
            └───────────────┘
                    │
                    ▼
              (Resume from here on crash)
```

#### 2. Channel Integration
OpenClaw connects to your existing communication platforms:
- **Telegram** - Direct bot commands
- **Discord** - Server-based interactions  
- **Slack** - Workspace integration
- **Signal** - Privacy-focused messaging
- **iMessage/WhatsApp** - Via extensions
- **Web** - Custom webhooks

Each channel has its own adapter that normalizes messages into a unified format.

#### 3. Model Providers
OpenClaw is model-agnostic but recommends:
- **Anthropic Claude** (Pro/Max) - Best for long-context reasoning
- **OpenAI GPT** - Codex for code tasks
- **Custom providers** - Plug in your own

#### 4. Skills System
OpenClaw has a skill system for extending capabilities:
- File system operations
- Git operations  
- Web search
- Code execution
- Custom skills via plugins

---

## How ScholarDevClaw Leverages OpenClaw

### The Wrapper Relationship

ScholarDevClaw wraps OpenClaw's infrastructure for a specific domain: **research-to-code automation**.

```
OpenClaw (Infrastructure)          ScholarDevClaw (Domain Application)
─────────────────────              ──────────────────────────────
Generic AI Assistant    ──────►    Research-Code Expert
Message-based UI        ──────►    CLI/TUI/API Workflows
General skills         ──────►    Research + Code skills
Basic context          ──────►    Deep codebase context
                       ──────►    Academic paper access
                       ──────►    Patch generation
                       ──────►    Validation pipelines
```

### What ScholarDevClaw Adds

| OpenClaw Base | ScholarDevClaw Enhancement |
|---------------|---------------------------|
| Gateway orchestration | Research-aware phase orchestration |
| Channel adapters | CLI + TUI + FastAPI interfaces |
| Basic skills | Repo analysis, paper extraction skills |
| Message context | Codebase embedding + research memory |
| Generic responses | Structured patches + validation |
| Simple tool use | Multi-step mapping → generation → validation |

### Integration Points

#### 1. Agent Bridge
ScholarDevClaw's TypeScript orchestrator (`agent/src/`) connects to the Python core:

```typescript
// agent/src/bridges/python-subprocess.ts
// Executes Python pipeline phases via structured JSON
const result = await pythonBridge.runPhase({
  phase: "analyze",
  repo: "/path/to/repo",
  options: { deep: true }
});
```

#### 2. HTTP Bridge  
For server deployments:
```typescript
// agent/src/bridges/python-http.ts
// Calls FastAPI endpoints
const response = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  body: JSON.stringify({ repo: "/path/to/repo" })
});
```

#### 3. State Persistence
Uses Convex (or local storage) for:
- Run history and checkpoints
- Approval decisions
- Integration results

---

## ScholarDevClaw Architecture Deep Dive

### The Six-Phase Pipeline

Every research-to-code task flows through these phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Pipeline Flow                                   │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │     1       │  REPO INTELLIGENCE
  │   Analyze   │  ─────────────────
  │             │  • Parse codebase structure
  └──────┬──────┘  • Detect languages/frameworks
         │         • Build dependency graph
         │         • Identify components
         ▼
  ┌─────────────┐
  │     2       │  RESEARCH INTELLIGENCE  
  │   Search    │  ─────────────────────
  │             │  • Search arXiv/PubMed/IEEE
  └──────┬──────┘  • Extract paper specs
         │         • Find related work
         ▼
  ┌─────────────┐
  │     3       │  MAPPING
  │    Map       │  ────────
  │             │  • Match research to code
  └──────┬──────┘  • Find target locations
         │         • Generate change plan
         ▼
  ┌─────────────┐
  │     4       │  PATCH GENERATION
  │  Generate   │  ─────────────────
  │             │  • Generate code changes
  └──────┬──────┘  • Create diff artifacts
         │         • Handle multi-file edits
         ▼
  ┌─────────────┐
  │     5       │  VALIDATION
  │  Validate   │  ────────────
  │             │  • Run tests
  └──────┬──────┘  • Syntax check
         │         • Security scan
         ▼         • Performance benchmark
  ┌─────────────┐
  │     6       │  REPORT
  │   Integrate │  ─────────
  │             │  • Show results
  └─────────────┘  • Apply to repo (optional)
```

### Core Modules

#### 1. Repo Intelligence (`repo_intelligence/`)
- **Tree-sitter analyzer** - Multi-language AST parsing
- **Dependency graph** - Import/dependency analysis  
- **Call graph** - Function/method relationships
- **Code embeddings** - Semantic similarity search
- **Refactoring** - Cross-file change planning

#### 2. Research Intelligence (`research_intelligence/`)
- **Paper sources** - arXiv, PubMed, IEEE Xplore APIs
- **Citation graph** - Paper relationship analysis
- **Similarity search** - TF-IDF + keyword matching
- **Spec extractor** - Extract algorithms from papers

#### 3. Mapping Engine (`mapping/`)
- Maps research specs to code targets
- Generates change proposals

#### 4. Patch Generation (`patch_generation/`)
- Generates precise code changes
- Multi-file coordination
- Diff artifact creation

#### 5. Validation (`validation/`)
- Test execution
- Syntax validation
- Security scanning (Bandit, Semgrep)
- Performance benchmarking

#### 6. Auth Module (`auth/`)
- API key management
- Encryption at rest
- Multi-profile support
- OAuth 2.0 flows
- Hardware key support (YubiKey)
- Team/role management

---

## Real-World Production Use Case

### Scenario: AI Startup Integrating Research

**Company**: NeuralFast AI (fictional)
**Product**: High-performance LLM inference engine
**Problem**: Want to integrate FlashAttention-3 but don't have time to read the paper

---

### The User Journey

#### Step 1: Initial Setup
```bash
# Sarah, the lead engineer, installs ScholarDevClaw
$ pip install -e ".[tui]"

# Configures authentication
$ scholardevclaw auth setup
✓ API key added (Anthropic)
✓ Encryption enabled

# Launches TUI
$ scholardevclaw tui
```

#### Step 2: Analysis
```bash
scholardevclaw > analyze ./neural-engine

[08:32:15] 🔍 Analyzing repository...
[08:32:16] ✓ Detected: Python (92%), PyTorch (88%)
[08:32:17] ✓ Found: 234 files, 1,847 functions, 156 classes
[08:32:18] ✓ Built: Dependency graph, Call graph
[08:32:19] ✓ Components: Attention (23), MLP (45), Embedding (12)

Results:
• Language: Python 3.10+ / PyTorch 2.0+
• Architecture: Transformer-based LLM
• Key files: attention.py, mlp.py, transformer.py
```

#### Step 3: Research Search
```bash
scholardevclaw > search flash attention optimization

Found 3 relevant papers:

[1] FlashAttention-3: Fast and Accurate Attention
    Authors: Tri Dao et al. (Stanford)
    arXiv: 2405.10973
    Impact: 2-4x speedup on H100 GPUs
    
[2] FlashDecoding++: Fast LLM Inference
    Authors: Shi et al.
    arXiv: 2403.01941
    Impact: 1.5-2x speedup
    
[3] Ring Attention: Long Context
    Authors: Liu et al.
    arXiv: 2310.07760
    Impact: 128K+ context support
```

#### Step 4: Map to Codebase
```bash
scholardevclaw > map flashattention-3

Mapping Analysis:
=================
Target: attention.py (847 lines)

Changes needed:
┌────────────────────────────────────────────┐
│ 1. Replace attention_forward()            │
│    → Use FlashAttention-3 kernel          │
│    → Location: lines 234-289              │
│                                             
│ 2. Add attention hardware detection       │
│    → Detect CUDA compute capability        │
│    → Fallback to SDPA for older GPUs      │
│                                             
│ 3. Update forward() signature              │
│    → Add flash_options parameter           │
└────────────────────────────────────────────┘

Confidence: 87%
Risk Level: MEDIUM
```

#### Step 5: Generate Patch
```bash
scholardevclaw > generate

Generating patch...
[08:45:01] ✓ Created: 3 file changes
[08:45:02] ✓ Files: attention.py, kernel_utils.py, __init__.py
[08:45:03] ✓ Lines: +156, -89

Patch Preview:
==============
- def attention_forward(x, ...):
+ def attention_forward(x, flash_options=None):
+     if flash_options and is_flash_available():
+         return flash_attention_3(x, **flash_options)
+     return scaled_dot_product_attention(x, ...)

[?] View full diff? (y/n)
```

#### Step 6: Validate
```bash
scholardevclaw > validate

Validation Suite:
=================
[1] Syntax Check.......... ✓ PASS (0 errors)
[2] Import Check......... ✓ PASS (all imports resolve)
[3] Type Check........... ✓ PASS (mypy clean)
[4] Unit Tests........... ✓ PASS (234/234)
[5] Integration Tests.... ✓ PASS (12/12)
[6] Security Scan........ ✓ PASS (Bandit: 0 issues)
[7] Performance.......... ✓ PASS
    Before: 1,247 ms/forward pass
    After:  892 ms/forward pass (+28.5% speedup)

[✓] All validations passed
```

#### Step 7: Integrate (Optional)
```bash
scholardevclaw > integrate --branch feature/flash-attention-3

[✓] Created branch: feature/flash-attention-3
[✓] Applied patch
[✓] Committed: "feat: Integrate FlashAttention-3 for 28% speedup"
[✓] Pushed to: origin/feature/flash-attention-3
[✓] Created PR: #247

Results Summary:
================
Files changed: 3
Lines added: 156
Lines removed: 89
Performance gain: +28.5%
Risk: MEDIUM (tested in CI)
```

---

### What Just Happened

Behind the scenes, ScholarDevClaw:

1. **Analyzed** the codebase → Built dependency + call graphs
2. **Searched** academic databases → Found relevant research
3. **Mapped** research to code → Identified exact change locations
4. **Generated** precise patches → Created production-ready code
5. **Validated** thoroughly → Ran tests, security scans, benchmarks
6. **Integrated** safely → Created branch, PR, all traceable

**Sarah got 28% performance improvement in 15 minutes** — without reading the paper, without manual implementation, without breaking anything.

---

## Technical Specifications

### Security (Production-Grade)

- **Encryption at rest**: Fernet + PBKDF2-HMAC-SHA256
- **API key management**: Hardware key support (YubiKey)
- **Audit logging**: All operations tracked
- **Rate limiting**: Per-key sliding window
- **Path traversal protection**: Regex validation
- **Atomic file operations**: No TOCTOU races

### Scalability

- **Async pipeline**: Full async/await throughout
- **Connection pooling**: Reusable HTTP connections
- **Circuit breakers**: Fail-fast for external calls
- **Caching**: Multi-level cache support

### Reliability

- **Rollback support**: One-click revert
- **Checkpoint persistence**: Resume from any phase
- **Health monitoring**: Live/readiness probes
- **Graceful shutdown**: Clean resource cleanup

---

## The Vision

ScholarDevClaw is building toward:

1. **Autonomous Research Engineering** — AI that finds, understands, and implements research autonomously
2. **Team Collaboration** — Shared context, approvals, rollbacks
3. **Enterprise Ready** — SSO, audit logs, compliance
4. **Plugin Ecosystem** — Custom validators, scanners, integrations

We're not just building another coding assistant. We're building the **infrastructure for operationalizing AI research** — because great research should reach production.

---

## Quick Start

```bash
# Install
cd core && pip install -e ".[tui]"

# Configure
scholardevclaw auth setup

# CLI usage
scholardevclaw analyze ./my-repo
scholardevclaw search "transformer normalization"
scholardevclaw map rmsnorm
scholardevclaw generate

# Or use the TUI
scholardevclaw tui

# Or API server
uvicorn scholardevclaw.api.server:app --reload
```

---

*ScholarDevClaw — Where Research Meets Production*
