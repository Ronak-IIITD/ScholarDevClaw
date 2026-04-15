## "Claude Code for Research" — Build Instructions for AI Agents

---

> **You are the primary engineering agent for ScholarDevClaw.**
> You have just cloned this repository for the first time.
> Read every word of this file before touching a single line of code.
> This is not a task list. This is the complete product vision, architecture,
> engineering contract, and execution playbook — all in one.
> Your job is to build something that does not exist yet:
> **a Claude Code-level agentic IDE, but for research scientists.**

---

## 0. What You Are Building

ScholarDevClaw is the **research-to-product engine** that serious AI researchers
have always needed and never had.

Today, a researcher reads a paper. They understand the idea. Then they spend
weeks bridging the gap between understanding and working code — debugging
environment setups, translating pseudocode, reproducing baselines, writing
boilerplate, packaging for others to use. That gap kills momentum. It turns
brilliant ideas into abandoned experiments.

ScholarDevClaw **closes that gap completely.**

The end-to-end promise:

```
researcher drops paper PDF (or DOI, or arXiv ID, or just a title)
         ↓
ScholarDevClaw reads it, understands it deeply
         ↓
generates a full, working, tested implementation from scratch
         ↓
benchmarks it against the paper's claimed results
         ↓
wraps it in an API, a demo UI, a Docker container, and a README
         ↓
researcher ships a prototype product the same day they read the paper
```

This is not an autocomplete tool. This is not a code search tool.
This is an **autonomous research engineer** that lives in the terminal.

The north star competitor is **Claude Code** — the gold standard for agentic
coding. We are building the equivalent for the research domain:
a tool that a PhD student, an ML engineer at a lab, or an ambitious
independent researcher reaches for the moment they finish reading a paper.

---

## 1. Product Philosophy

### 1.1 Principles — never compromise these

**Depth over breadth.** It is better to flawlessly implement one paper
than to partially implement ten. Every feature must work end-to-end before
a new one begins.

**Researchers are the users, not developers.** A researcher should not need
to understand your internal architecture to use this tool. The UX must be
as clean as `git clone`.

**Running code is the only truth.** Generating code that looks right is
not enough. Every generated file must execute. Every claimed metric must
be verified by actually running the implementation.

**Local-first, cloud-optional.** The tool must work completely offline
with no API keys for its core function. LLM features are enhancements,
not requirements. A researcher at an air-gapped lab must be able to use this.

**Reproducibility is sacred.** Research is meaningless without it.
Every output ScholarDevClaw produces must be traceable back to the exact
paper version, exact model weights, exact random seeds, and exact
environment that produced it.

**Never surprise the researcher.** Show every action before taking it.
Show confidence scores. Show diffs. Require approval for anything destructive.
A bad patch to a research codebase can corrupt months of experiment history.

### 1.2 The Two Modes

ScholarDevClaw operates in two distinct modes. Both must be world-class.

**Mode A — Paper-to-Implementation (from scratch)**
The researcher has no existing code. They drop a paper and want a working
implementation. ScholarDevClaw builds the entire project from zero.

**Mode B — Codebase-to-Improvement (patch existing code)**
The researcher has a working codebase and wants to apply findings from
new papers to it. ScholarDevClaw analyzes their code, finds applicable
improvements from relevant papers, generates validated patches.

The existing `v2` codebase handles Mode B partially.
Mode A is the primary new buildout. Both must be excellent.

---

## 2. Repository Map

```
ScholarDevClaw/
├── core/                          # PRIMARY — Python package, CLI, all agents
│   ├── scholardevclaw/
│   │   ├── cli.py                 # unified CLI entry point (Click)
│   │   ├── analyzer.py            # existing codebase analyzer (Mode B)
│   │   ├── generator.py           # existing patch generator (Mode B)
│   │   ├── validator.py           # existing validator (Mode B)
│   │   ├── specs/                 # existing paper specs (Mode B, keep)
│   │   │
│   │   ├── ingestion/             # NEW — PDF/DOI/arXiv ingestion
│   │   ├── understanding/         # NEW — deep paper comprehension
│   │   ├── planning/              # NEW — dynamic implementation planning
│   │   ├── generation/            # NEW — multi-agent code generation
│   │   ├── execution/             # NEW — sandboxed execution + healing
│   │   ├── product/               # NEW — API, demo, docs scaffold
│   │   ├── knowledge/             # NEW — vector store, pattern library
│   │   ├── tui/                   # EXISTING — extend, don't rewrite
│   │   └── workspace/             # NEW — session state management
│   │
│   ├── tests/
│   └── pyproject.toml
│
├── agent/                         # orchestration layer (extend)
├── web/                           # TypeScript frontend (extend)
├── convex/                        # backend state
├── docker/
│   ├── sandbox.Dockerfile         # NEW — isolated execution environment
│   └── docker-compose*.yml        # existing
├── landing/                       # GitHub Pages site (already redesigned)
└── docs/
```

**Filesystem rules:**
- All new Python modules go inside `core/scholardevclaw/`
- Never move or rename existing files without a migration note in the PR
- All tests mirror the source structure exactly:
  `core/scholardevclaw/ingestion/pdf_parser.py`
  → `core/tests/test_ingestion_pdf_parser.py`
- New CLI commands are added to `core/scholardevclaw/cli.py`, never in
  separate files

---

## 3. Complete Architecture

### 3.1 The Seven-Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 7  │  INTERFACES                                         │
│           │  CLI (Click) · TUI (Textual) · Web (Next.js) · API  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 6  │  PRODUCT SCAFFOLD                                   │
│           │  FastAPI wrapper · Gradio demo · PyPI package ·     │
│           │  Dockerfile · README · GitHub Actions CI            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5  │  EXECUTION ENGINE                                   │
│           │  Docker sandbox · pytest harness · self-healing     │
│           │  loop · reproducibility scorer · metric extractor   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4  │  CODE GENERATION                                    │
│           │  Orchestrator agent · parallel module agents ·      │
│           │  dependency-aware task graph · syntax validator ·   │
│           │  integration healer                                 │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3  │  IMPLEMENTATION PLANNER                             │
│           │  Dynamic module decomposition · tech stack selector │
│           │  · dependency graph builder · test strategy gen     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2  │  PAPER UNDERSTANDING                                │
│           │  LLM comprehension agent · concept graph builder ·  │
│           │  requirement extractor · domain classifier          │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1  │  INGESTION                                          │
│           │  PDF parser · arXiv fetcher · DOI resolver ·        │
│           │  equation extractor · algorithm detector · OCR      │
├─────────────────────────────────────────────────────────────────┤
│  CROSS    │  KNOWLEDGE BASE                                     │
│  CUTTING  │  ChromaDB vector store · embedding engine ·         │
│           │  paper index · implementation pattern library ·     │
│           │  session state · workspace manager                  │
└─────────────────────────────────────────────────────────────────┘
```

Every layer communicates **only through typed dataclass interfaces**.
No layer reaches past its immediate neighbor.
No global state except the workspace manager.

### 3.2 Data Flow — Mode A (Paper to Implementation)

```
Input: arxiv:1706.03762
   │
   ▼ Layer 1: Ingestion
PaperDocument {
  title, abstract, authors, year
  sections: [Section]
  equations: [Equation]     ← LaTeX extracted
  algorithms: [Algorithm]   ← pseudocode blocks
  figures: [Figure]         ← captions + images
  references: [str]
  domain: "nlp"
}
   │
   ▼ Layer 2: Understanding
PaperUnderstanding {
  one_line_summary: str
  key_insight: str
  contributions: [Contribution]
  requirements: [Requirement]   ← datasets, libs, hardware
  concept_nodes: [ConceptNode]  ← knowledge graph
  concept_edges: [ConceptEdge]
  core_algorithm: str           ← plain English
  input_output_spec: str
  evaluation_protocol: str
  complexity: "medium"
  confidence: 0.87
}
   │
   ▼ Layer 3: Planning
ImplementationPlan {
  project_name: "attention_is_all_you_need"
  tech_stack: "pytorch"
  modules: [CodeModule]     ← topologically sorted
  directory_structure: dict
  environment: {"torch": ">=2.0"}
  entry_points: ["train", "evaluate"]
  estimated_lines: 1840
}
   │
   ▼ Layer 4: Generation (parallel async)
GenerationResult {
  module_results: [ModuleResult]  ← one per CodeModule
  output_dir: Path
  success_rate: 0.91
  total_tokens: 48200
  duration_seconds: 142
}
   │
   ▼ Layer 5: Execution
ExecutionReport {
  tests_passed: 47
  tests_failed: 3
  exit_code: 1
  stdout: "..."
  stderr: "ImportError: No module named..."
} ──── healing loop ────► re-generate failing modules
   │                      ↑_________________________│
   ▼ (after healing)
ReproducibilityReport {
  claimed_metrics: {"BLEU": 28.4, "perplexity": 4.92}
  achieved_metrics: {"BLEU": 27.8, "perplexity": 5.01}
  score: 0.97
  verdict: "reproduced"
}
   │
   ▼ Layer 6: Product
project/
  src/           ← full implementation
  tests/         ← 47 passing tests
  api/main.py    ← FastAPI service
  demo.py        ← Gradio UI
  Dockerfile
  pyproject.toml
  README.md      ← with reproducibility table
  .github/workflows/ci.yml
```

### 3.3 Data Flow — Mode B (Codebase Improvement, existing)

```
Input: ./my-transformer-repo + optional paper query
   │
   ▼ Analyze (existing analyzer.py)
CodebaseProfile {
  language: "python"
  frameworks: ["pytorch"]
  patterns: [CodePattern]     ← detected optimization opportunities
  ast_summary: dict
}
   │
   ▼ Search (existing + enhanced)
PaperMatches {
  matches: [PaperMatch]       ← ranked by relevance to detected patterns
}
   │
   ▼ Map (enhanced 6-tier matching)
MappingResult {
  mappings: [CodeMapping]     ← paper concept → code location
  confidence_scores: dict
}
   │
   ▼ Generate + Validate (existing + enhanced)
PatchArtifact {
  diff: str
  benchmark_delta: float
  test_results: TestResults
  pr_description: str
}
```

---

## 4. Module Specifications

### 4.1 `ingestion/` — Layer 1

**`models.py`** — define these dataclasses exactly:

```python
@dataclass
class Equation:
    latex: str          # raw LaTeX, never converted
    description: str    # ≤200 chars of surrounding text
    page: int
    equation_type: str  # "loss" | "model" | "metric" | "notation" | "unknown"

@dataclass
class Algorithm:
    name: str           # "Algorithm 1: Scaled Dot-Product Attention"
    pseudocode: str     # verbatim, preserving indentation
    page: int
    language_hint: str  # "python-like" | "math" | "c-like" | "unknown"
    inputs: list[str]   # extracted parameter names
    outputs: list[str]

@dataclass
class Figure:
    caption: str
    page: int
    figure_type: str    # "architecture" | "results" | "diagram" | "plot"
    image_path: Optional[Path]

@dataclass
class Section:
    title: str
    level: int          # 1=H1, 2=H2, 3=H3
    content: str
    page_start: int
    section_type: str   # "introduction" | "method" | "experiments" | "conclusion" | "related"

@dataclass
class PaperDocument:
    # identity
    title: str
    authors: list[str]
    arxiv_id: Optional[str]
    doi: Optional[str]
    year: Optional[int]
    abstract: str
    venue: Optional[str]     # "NeurIPS 2017", "ICML 2023"

    # structured content
    sections: list[Section]
    equations: list[Equation]
    algorithms: list[Algorithm]
    figures: list[Figure]
    tables: list[dict]       # raw extracted tables

    # raw
    full_text: str
    pdf_path: Optional[Path]
    source_url: Optional[str]

    # classification
    references: list[str]
    keywords: list[str]
    domain: str              # "cv" | "nlp" | "rl" | "systems" | "theory" | "biology" | "multimodal"
    subdomain: str           # "object-detection" | "language-modeling" | "policy-gradient" | etc.

    # serialization
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "PaperDocument": ...
```

**`pdf_parser.py`** — `PDFParser` class:
- Primary: `pymupdf` (fitz) for text extraction
- Fallback: `pdfplumber` for table and layout extraction
- Figure extraction: use `fitz.Document.get_images()`, save PNG to
  `{work_dir}/figures/fig_{page}_{idx}.png`
- Algorithm detection: look for blocks matching patterns:
  `Algorithm \d+`, `Procedure \d+`, lines with `Input:` / `Output:` / `for` / `while`
- Equation detection: LaTeX delimiters `$...$`, `$$...$$`,
  `\begin{equation}`, `\begin{align}`, `\begin{gather}`
- Domain classifier (keyword sets):
  - `nlp`: transformer, bert, gpt, attention, tokeniz*, language model
  - `cv`: convolution, resnet, yolo, detection, segmentation, vit
  - `rl`: reward, policy, q-learning, environment, agent, markov
  - `systems`: kernel, mutex, scheduler, memory, cache, network
  - `theory`: theorem, proof, lemma, corollary, bound, complexity
  - `biology`: protein, sequence, genomics, rna, dna, cell
  - `multimodal`: vision-language, clip, image-text, audio-visual
- Section type classifier: match title against known patterns

**`paper_fetcher.py`** — `PaperFetcher` class:
- `fetch_auto(source: str, dest: Path) -> PaperDocument`
  - detect source type: starts with `arxiv:` → `fetch_by_arxiv_id`
  - starts with `10.` → `fetch_by_doi`
  - starts with `http` → `fetch_by_url`
  - otherwise → `search_by_title`
- `fetch_by_arxiv_id`: use `arxiv` library, download PDF, parse, merge metadata
- `fetch_by_doi`: Semantic Scholar API `https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}`
  - fields: `title,authors,year,abstract,openAccessPdf,venue,externalIds`
- `fetch_by_url`: HEAD request, detect content-type, follow PDF links in HTML
- `search_by_title`: arXiv search + Semantic Scholar search, take top result
  above 0.85 title similarity (use `difflib.SequenceMatcher`)
- All fetchers: 3 retries with exponential backoff, 30s timeout per request
- Cache: save to `~/.scholardevclaw/cache/{arxiv_id}.json` to avoid re-fetching

**CLI command:**
```
scholardevclaw ingest <source> [--output-dir DIR] [--no-cache] [--verbose]
```

### 4.2 `understanding/` — Layer 2

**`models.py`** — define exactly:

```python
@dataclass
class Contribution:
    claim: str              # ≤1 sentence
    novelty: str            # what is new vs prior work
    is_implementable: bool  # can this be coded right now?
    implementation_notes: str  # hints for the generator

@dataclass
class Requirement:
    name: str               # "ImageNet-1K", "CUDA 11.8", "PyTorch ≥2.0"
    requirement_type: str   # "dataset" | "library" | "hardware" | "baseline" | "pretrained_model"
    is_optional: bool
    version_constraint: Optional[str]
    acquisition_url: Optional[str]
    notes: str

@dataclass
class ConceptNode:
    id: str                 # snake_case unique
    label: str
    concept_type: str       # "model" | "operation" | "loss" | "dataset" | "metric" | "technique"
    description: str        # ≤100 words
    paper_section: str      # which section defines this

@dataclass
class ConceptEdge:
    source_id: str
    target_id: str
    relation: str           # "uses" | "produces" | "replaces" | "compared_against" | "trained_on" | "evaluated_on"
    weight: float           # 0.0–1.0, how central is this relation

@dataclass
class PaperUnderstanding:
    paper_title: str
    one_line_summary: str          # ≤20 words
    problem_statement: str         # what problem this solves
    prior_state_of_art: str        # what existed before
    key_insight: str               # the core idea in 2–3 sentences
    why_it_works: str              # the mechanism, in plain English

    contributions: list[Contribution]
    requirements: list[Requirement]

    concept_nodes: list[ConceptNode]
    concept_edges: list[ConceptEdge]

    core_algorithm_description: str   # plain English, no jargon, step-by-step
    input_output_spec: str            # "takes X tensors of shape Y, produces Z"
    hyperparameters: dict             # name → default value from paper
    evaluation_protocol: str          # datasets, metrics, how measured
    known_limitations: str            # from the paper itself or obvious ones

    complexity: str                   # "trivial" | "low" | "medium" | "high" | "frontier-only"
    estimated_impl_hours: int         # rough but honest estimate
    can_reproduce_without_compute: bool  # can a 24GB GPU card reproduce this?
    confidence: float                 # agent's self-assessed confidence 0.0–1.0
    confidence_notes: str             # what the agent is uncertain about
```

**`agent.py`** — `UnderstandingAgent`:

System prompt (insert verbatim):
```
You are a world-class AI researcher and senior software engineer with deep
expertise across machine learning, computer science, and scientific writing.
You read research papers with surgical precision and extract structured
information that allows a developer to implement the paper completely from
scratch — without reading the paper themselves.

Your job is ANALYSIS, not summarization. You are not writing an abstract.
You are reverse-engineering the paper into an implementation blueprint.

Critical rules:
1. Never hallucinate results, citations, or metrics not stated in the paper.
2. If you are unsure about something, say so explicitly in confidence_notes.
3. The core_algorithm_description must be step-by-step, implementation-ready,
   and understandable by someone who has never read the paper.
4. Requirements must be exhaustive — missing a required dataset or library
   wastes the user's time.
5. Respond with valid JSON only. No markdown. No explanation outside the JSON.
```

Prompt construction strategy:
- Include: title, abstract, all algorithm blocks (verbatim), top 20 equations
  with context, method/model section, experiments section (for eval protocol),
  conclusion
- Exclude: related work (too noisy), acknowledgments, appendices (unless
  they contain the main algorithm)
- Token budget: stay under 80k tokens; truncate sections in reverse order
  of importance (related work first, conclusion last)
- If paper is > 80k tokens: split into two calls — architecture pass,
  then experiments pass — merge results

**`graph.py`** — build `networkx.DiGraph` from `PaperUnderstanding`,
export to `concept_graph.json` using `nx.node_link_data()`.
Include graph metrics: density, key hubs (highest in-degree nodes),
longest path (implementation order hint).

**CLI command:**
```
scholardevclaw understand <paper_document.json|dir> [--model MODEL] [--output-dir DIR]
```

### 4.3 `planning/` — Layer 3

**`models.py`**:

```python
@dataclass
class CodeModule:
    id: str                      # snake_case, unique in plan
    name: str
    description: str             # what this module does (≤100 words)
    file_path: str               # relative to project root
    module_type: str             # "data" | "model" | "training" | "evaluation" | "utils" | "api" | "demo"
    depends_on: list[str]        # other module ids — must form a DAG
    priority: int                # topological level (1 = no dependencies)
    estimated_lines: int
    test_file_path: str
    tech_stack: str              # "pytorch" | "jax" | "numpy" | "fastapi" | "stdlib"
    key_classes: list[str]       # class names to implement
    key_functions: list[str]     # top-level functions to implement
    paper_sections: list[str]    # which paper sections this implements
    complexity: str              # "trivial" | "low" | "medium" | "high"

@dataclass
class ImplementationPlan:
    project_name: str            # valid Python package name, snake_case
    target_language: str         # "python" (always for now)
    tech_stack: str              # primary stack
    python_version: str          # "3.11"
    modules: list[CodeModule]    # sorted by priority ascending
    directory_structure: dict    # nested dict representing full file tree
    environment: dict            # {package_name: version_constraint}
    dev_environment: dict        # test/dev deps
    entry_points: list[str]      # module ids that are user-facing
    estimated_total_lines: int
    implementation_order: list[str]  # module ids in exact build order
    notes: str                   # anything the generator should know
```

**`planner.py`** — `ImplementationPlanner`:

Tech stack selection (deterministic, no LLM):
- Scan `understanding.requirements` for library names
- `jax` in requirements → `jax`
- `tensorflow` in requirements, no torch → `tensorflow`
- domain is `systems` and no DL lib in requirements → `numpy-only`
- domain is `biology` and `biopython`/`rdkit` in requirements → `bio`
- default → `pytorch` (correct for 90% of ML papers)

Planning LLM call — the prompt must include:
- Paper summary, core algorithm, requirements, evaluation protocol
- The selected tech stack
- The full concept graph (node list with types)
- Complexity estimate from understanding
- Explicit instruction: "Every paper needs at minimum a data loader,
  model definition, training loop, evaluation harness, and README generator.
  No placeholder modules — every module must be implementable in one pass."

Validation after planning:
- Assert modules form a valid DAG (no cycles) using `networkx.is_dag()`
- Assert every `depends_on` reference exists in the module list
- Assert every module has a corresponding test file path
- Assert priority ordering is consistent with dependency graph
- Raise `InvalidPlanError` with specific message if any assertion fails

**CLI command:**
```
scholardevclaw plan <understanding.json|dir> [--stack STACK] [--output-dir DIR] [--dry-run]
```

### 4.4 `generation/` — Layer 4

This is the most complex layer. Read carefully.

**`module_agent.py`** — `ModuleAgent`:

The module agent generates one `CodeModule`. It runs in async context.

System prompt:
```
You are a world-class Python engineer implementing a research paper module.
Write production-quality, fully typed Python 3.11 code.

Non-negotiable rules:
1. Every public class and function has a Google-style docstring.
2. All function signatures have type hints including return types.
3. Imports are ordered: stdlib → third-party → local. One blank line between groups.
4. No placeholder logic. No "# TODO: implement". Write the real implementation.
5. If implementing a neural network: use proper weight initialization,
   follow the paper's architectural choices exactly, and include forward()
   with shape comments: # (batch, seq_len, d_model)
6. Match the paper's variable names where reasonable (d_model, n_heads, etc.)
7. Code must be importable with zero modifications.
8. Return ONLY Python source code. No markdown. No explanation.
```

Generation strategy:
- Context injection: for each `depends_on` module, inject up to 2000 chars
  of its implementation as context
- Knowledge base injection: retrieve top-3 similar past implementations
  from ChromaDB, inject as examples (labeled "# Reference implementation:")
- Paper context injection: inject the specific sections of the paper that
  this module implements, plus relevant equations and algorithm blocks
- Self-healing: on SyntaxError, re-prompt with exact error and line number
- Max 3 attempts per module — if still failing, mark as `partial` and continue

Test generation:
```
You are an expert in pytest and scientific computing.
Write thorough tests for this research implementation module.

Rules:
1. Use pytest fixtures for all shared setup.
2. Mock external calls (network, GPU) with monkeypatch.
3. Test tensor shapes explicitly: assert output.shape == (batch_size, seq_len, d_model)
4. Test numerical properties: assert output.dtype == torch.float32
5. Test edge cases: empty input, single element, max sequence length.
6. Include at least one integration test that runs the full forward pass.
7. Return ONLY pytest code. No markdown.
```

**`orchestrator.py`** — `CodeOrchestrator`:

```python
async def generate_all(
    self,
    plan: ImplementationPlan,
    understanding: PaperUnderstanding,
    doc: PaperDocument,
    output_dir: Path,
    knowledge_base: Optional[KnowledgeBase] = None,
    max_parallel: int = 4,
    progress_callback: Optional[Callable] = None,
) -> GenerationResult:
```

Execution model:
- Group modules by `priority` level
- Within each level, generate all modules in parallel (capped by `max_parallel`)
- After each level completes, modules become available as context for the next
- Use `asyncio.Semaphore(max_parallel)` for concurrency control
- Track token usage, timing, and attempt counts per module
- Write each completed module to disk immediately (don't batch)
- Emit progress events for the TUI to consume

File writing rules:
- Create all parent directories before writing
- Write `.py` files with UTF-8 encoding
- Write `__init__.py` in every package directory
- Generate a `requirements.txt` from the plan's `environment` dict
- Generate `setup.cfg` and `pyproject.toml`

**CLI command:**
```
scholardevclaw generate <plan.json|dir> <understanding.json|dir> \
    [--output-dir DIR] [--max-parallel N] [--model MODEL] [--no-kb]
```

### 4.5 `execution/` — Layer 5

**`sandbox.py`** — `SandboxRunner`:

Docker configuration:
- Base image: `python:3.11-slim`
- No network access inside container: `network_mode="none"`
- Memory limit: configurable, default 4GB
- CPU limit: 2 cores
- Timeout: configurable, default 300s
- Working directory: `/workspace` (mounted from `project_dir`)
- The container must never write outside `/workspace` and `/tmp`

`sandbox.Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Pre-install common ML deps at build time
RUN pip install --no-cache-dir \
    pytest pytest-json-report pytest-cov \
    torch --index-url https://download.pytorch.org/whl/cpu \
    jax[cpu] flax optax \
    numpy scipy scikit-learn pandas \
    transformers datasets tokenizers \
    fastapi uvicorn gradio \
    networkx matplotlib seaborn

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src
CMD ["pytest", "tests/", "-v", "--json-report", "--json-report-file=/tmp/report.json"]
```

Build this image once: `docker build -f docker/sandbox.Dockerfile -t sdc-sandbox:latest .`
Add to `scripts/runbook.sh` under `dev setup`.

Test result parsing:
- Extract `/tmp/report.json` from container after run
- Parse pytest-json-report format for per-test results
- Identify exactly which test files and test functions failed
- Map failing test files back to source modules via naming convention

**`healer.py`** — `SelfHealingLoop`:

Healing strategy:
1. Run tests → get `ExecutionReport`
2. Parse failures → identify failing module ids (via test file → module mapping)
3. Extract the specific error messages per module from stderr
4. Re-invoke `ModuleAgent` with the original prompt PLUS the error context:
   ```
   Previous implementation had these runtime errors:
   {stderr excerpt for this module}
   Fix all errors. The test that failed was: {test_function_name}
   The test expects: {test_source_code}
   ```
5. Write healed module, re-run only the failing tests
6. Repeat up to `max_rounds` times (default: 3)
7. After max rounds, mark remaining failures as `unhealed` and continue

**`scorer.py`** — `ReproducibilityScorer`:

Metric extraction from stdout:
- LLM call: "Extract all numeric metric results from this output text.
  Return JSON: {metric_name: value}. Common patterns: 'Accuracy: X%',
  'BLEU: X', 'perplexity: X', 'loss: X.XX'"
- Normalize metric names: `acc` → `accuracy`, `ppl` → `perplexity`

Scoring formula:
- For each claimed metric, compute `ratio = min(achieved/claimed, claimed/achieved)`
- Weighted average by metric importance (accuracy/main metric > secondary metrics)
- Score < 0.5 → "failed", 0.5–0.9 → "partial", ≥ 0.9 → "reproduced"
- If no claimed metrics found → score = 0.5, verdict = "unverifiable"

**CLI command:**
```
scholardevclaw execute <project_dir> [--heal] [--rounds N] [--timeout N] [--output-dir DIR]
```

### 4.6 `product/` — Layer 6

**`scaffolder.py`** — `ProductScaffolder`:

Uses Jinja2 templates. All templates live in `product/templates/`.

Templates to implement (every one must produce real, runnable code):

`api_main.py.j2` → `api/main.py`:
- FastAPI app with `/predict`, `/health`, `/docs` routes
- Pydantic input/output models derived from paper's I/O spec
- Proper error handling (422 for bad input, 500 for model error)
- Async endpoints
- CORS middleware enabled

`gradio_demo.py.j2` → `demo.py`:
- Gradio interface appropriate to domain:
  - `nlp`: text input → text output
  - `cv`: image upload → image/text output
  - `rl`: environment visualization + action output
- Example inputs loaded from paper's experiments section
- Clear description and paper citation in the interface

`pyproject.toml.j2` → `pyproject.toml`:
- All deps from plan's environment dict
- Dev deps: pytest, black, mypy, ruff
- Entry points for CLI if applicable
- Build system: `hatchling`

`Dockerfile.j2` → `Dockerfile`:
- Multi-stage: builder (with dev deps) → runtime (minimal)
- Non-root user
- Health check endpoint

`README.md.j2` → `README.md`:
Must include all of:
- Paper title, authors, venue, link
- One-line description
- Key insight (plain English)
- Architecture diagram (ASCII art generated from concept graph)
- Reproducibility table: claimed vs achieved metrics
- Install instructions (pip + Docker)
- Quickstart: train, evaluate, demo, API
- Configuration: all hyperparameters with defaults from paper
- Citation block (BibTeX)
- License

`ci.yml.j2` → `.github/workflows/ci.yml`:
- Python 3.11 matrix
- Install deps
- Run pytest
- Run mypy
- Run ruff linter
- Docker build check

**CLI command:**
```
scholardevclaw scaffold <project_dir> [--plan PLAN] [--understanding UNDERSTANDING] \
    [--repro REPRO_REPORT] [--output-dir DIR]
```

### 4.7 `knowledge/` — Cross-cutting

**`store.py`** — `KnowledgeBase`:

Storage: ChromaDB persistent client at `~/.scholardevclaw/kb/`
Embedding model: `BAAI/bge-small-en-v1.5` (33M params, runs on CPU, excellent quality)

Three collections:
- `papers`: one document per paper, metadata includes domain, complexity, arxiv_id
- `implementations`: one document per generated module, metadata includes module_type, tech_stack
- `patterns`: curated high-quality code patterns (seeded from existing `specs/`)

`store_paper(doc, understanding)`:
- Embed: `f"{doc.title}. {doc.abstract}. {understanding.key_insight}"`
- Store full understanding JSON as metadata (up to 16kb limit)

`store_implementation(module, code, understanding)`:
- Embed: `f"{module.name}: {module.description}\n{code[:500]}"`
- Store: module_type, tech_stack, paper_title, domain, lines_of_code

`retrieve_similar_implementations(description, tech_stack, n=3) -> list[str]`:
- Query by description embedding
- Filter by tech_stack
- Return code strings (context for generation)

`retrieve_similar_papers(query, domain=None, n=5) -> list[dict]`:
- Return papers with their understanding summaries

**`workspace.py`** — `WorkspaceManager`:

Manages session state. One workspace per `from-paper` invocation.

```python
@dataclass
class WorkspaceSession:
    session_id: str          # UUID
    source: str              # original input (arxiv ID, PDF path, etc.)
    created_at: datetime
    status: str              # "ingesting" | "understanding" | "planning" |
                             # "generating" | "executing" | "scaffolding" | "done" | "failed"
    current_phase: int       # 1–6
    paper_document: Optional[PaperDocument]
    understanding: Optional[PaperUnderstanding]
    plan: Optional[ImplementationPlan]
    generation_result: Optional[GenerationResult]
    execution_report: Optional[ExecutionReport]
    reproducibility_report: Optional[ReproducibilityReport]
    output_dir: Path
    error: Optional[str]
    checkpoints: list[str]   # phases that completed successfully

class WorkspaceManager:
    def create_session(self, source: str, output_dir: Path) -> WorkspaceSession: ...
    def save_checkpoint(self, session: WorkspaceSession, phase: int) -> None: ...
    def resume_session(self, session_id: str) -> WorkspaceSession: ...
    def list_sessions(self) -> list[WorkspaceSession]: ...
```

Sessions persist to `~/.scholardevclaw/sessions/{session_id}.json`.
Resumable: if a session failed at phase 4, `from-paper --resume {id}` restarts from phase 4.

---

## 5. The Unified `from-paper` Command

This is the product. Everything else serves this.

```
scholardevclaw from-paper <source>
  [--output-dir DIR]       default: ./<project_name>/
  [--heal]                 enable self-healing execution loop
  [--scaffold]             generate API + demo + docs
  [--max-parallel N]       default: 4 generation agents
  [--model MODEL]          default: claude-opus-4-5
  [--no-kb]                skip knowledge base
  [--dry-run]              show plan only, no code generation
  [--resume SESSION_ID]    resume a previous failed session
  [--watch]                stream live progress in terminal
  [--export ZIP]           package output as a zip file
```

Progress output format (required):
```
ScholarDevClaw — paper to product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Paper    Attention Is All You Need (Vaswani et al., 2017)
  Domain   NLP → Language Modeling
  Stack    PyTorch 2.0+

  [1/6] Ingesting...                             ✓  0.8s
  [2/6] Understanding...                         ✓  12.4s
  [3/6] Planning...          9 modules          ✓  8.2s
  [4/6] Generating...        ████████░░  7/9   ↻  42.1s
        ├── data_loader      ✓
        ├── tokenizer        ✓
        ├── positional_enc   ✓
        ├── attention        ✓
        ├── transformer      ✓
        ├── train_loop       ✓
        ├── evaluate         ✓
        ├── readme_gen       ↻ generating...
        └── inference        ○ queued
  [5/6] Executing...
  [6/6] Scaffolding...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Output   ./attention_is_all_you_need/
  Tests    47 passing / 0 failing
  Repro    97.3% (reproduced)
  Time     2m 41s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 6. TUI Extension

The existing Textual TUI must be extended with:

**New screens (add, don't replace existing):**

`PaperIngestionScreen`:
- Input: paste arXiv ID, DOI, URL, or drag-drop PDF path
- Shows parsed paper metadata as it extracts
- Preview panel: list of detected algorithms + equations

`UnderstandingScreen`:
- Shows the concept graph as an ASCII art network
- Lists contributions, requirements, complexity estimate
- Confidence meter with expandable notes
- "Looks good → proceed" or "Edit understanding" actions

`PlanningScreen`:
- Shows module dependency graph (ASCII tree)
- Estimated lines, time, token cost per module
- Tech stack display with justification
- Approval gate: user must confirm before generation starts

`GenerationScreen`:
- Live log streaming per module (switchable panes)
- Progress bar per module, overall progress
- Syntax error alerts inline
- Token usage counter

`ExecutionScreen`:
- Live pytest output
- Per-test pass/fail status
- Healing round indicator
- Reproducibility score build-up

`ProductScreen`:
- File tree of generated project
- Preview pane for any file
- "Open in editor" button (respects `$EDITOR`)
- One-click copy install command

---

## 7. Engineering Standards

### 7.1 Code Quality — non-negotiable

Every file must pass all of these before committing:
```bash
ruff check core/scholardevclaw/          # zero warnings
mypy --strict core/scholardevclaw/       # zero errors
pytest core/tests/ -v --cov=scholardevclaw --cov-report=term-missing
# coverage must be ≥ 60% per new module
```

Pre-commit hooks must run all three. Add to `.pre-commit-config.yaml`.

### 7.2 Type system

- Python 3.11+ throughout. Use `from __future__ import annotations` in every file.
- All dataclasses use `@dataclass(frozen=True)` unless mutation is required.
- All collections are typed: `list[str]`, `dict[str, float]`, not bare `list`/`dict`.
- Never use `Any` except in JSON parsing functions, and mark those with `# type: ignore[assignment]`.
- Use `Optional[X]` (not `X | None`) for Python 3.10 compatibility in the public API.

### 7.3 Error handling

```python
# WRONG
try:
    result = do_thing()
except Exception:
    pass

# WRONG
try:
    result = do_thing()
except Exception as e:
    print(e)

# RIGHT
try:
    result = do_thing()
except SpecificError as e:
    logger.error("Failed to do thing: %s", e)
    raise ScholarDevClawError(f"Could not do thing: {e}") from e
```

Define a clear exception hierarchy in `core/scholardevclaw/exceptions.py`:
```python
class ScholarDevClawError(Exception): ...
class IngestionError(ScholarDevClawError): ...
class PaperNotAccessibleError(IngestionError): ...
class UnderstandingError(ScholarDevClawError): ...
class PlanningError(ScholarDevClawError): ...
class GenerationError(ScholarDevClawError): ...
class ExecutionError(ScholarDevClawError): ...
class SandboxError(ExecutionError): ...
class KnowledgeBaseError(ScholarDevClawError): ...
```

### 7.4 Logging

```python
import logging
logger = logging.getLogger(__name__)   # one per module
```

- `DEBUG`: LLM prompt content, full API responses, per-file operations
- `INFO`: phase transitions, major decisions (stack selection, module count)
- `WARNING`: degraded path taken (fallback parser, KB miss, partial generation)
- `ERROR`: failure that needs user attention but is recoverable
- `CRITICAL`: unrecoverable failure requiring process exit

Never use `print()` except in CLI output functions (those use `click.echo()`).

### 7.5 LLM calls

All LLM calls must:
- Have an explicit timeout (120s default)
- Retry 3 times with exponential backoff (1s, 2s, 4s) on rate limit / server error
- Log token usage at DEBUG level
- Return a typed dataclass, never raw strings
- Strip markdown fences before JSON parsing:
  ```python
  def clean_json_response(raw: str) -> dict:
      cleaned = raw.strip()
      if cleaned.startswith("```"):
          cleaned = cleaned.split("```")[1]
          if cleaned.startswith("json"):
              cleaned = cleaned[4:]
      return json.loads(cleaned.strip())
  ```

### 7.6 Testing requirements

Every new module must have:
- At least one unit test per public function
- At least one integration test (marked `@pytest.mark.integration`, skipped in CI)
- Fixtures in `conftest.py` for common test data (paper documents, understandings, plans)
- Mocking of all external calls (Anthropic API, arXiv, Docker SDK)
- Shape assertions for any tensor-producing code
- Type correctness assertions (check returned types match annotations)

Fixture files to create in `core/tests/fixtures/`:
- `attention_paper_document.json` — PaperDocument for Attention Is All You Need
- `attention_understanding.json` — PaperUnderstanding for the same
- `simple_plan.json` — a 3-module plan for testing generation
- `broken_module.py` — a Python file with known syntax errors (for healer tests)
- `passing_project/` — minimal project that passes all tests (for sandbox tests)

### 7.7 CLI UX requirements

- Every command that takes > 2 seconds shows a spinner or progress bar
- Use `click.echo(message, err=True)` for all error output
- All error messages must include: what failed, why, and what to do about it
- Exit codes: 0=success, 1=partial (some modules failed), 2=complete failure, 3=config error
- Environment variable for every configurable: `ANTHROPIC_API_KEY`, `SDC_MODEL`,
  `SDC_KB_DIR`, `SDC_SANDBOX_IMAGE`, `SDC_MAX_PARALLEL`, `SDC_LOG_LEVEL`
- `--help` on every command shows real examples, not just flag descriptions

### 7.8 Security

- Never log API keys, even at DEBUG level
- Sandbox must have `network_disabled=True` in Docker SDK call
- Validate all file paths before writing: raise if path escapes the output directory
- Never execute generated code outside the Docker sandbox
- Rate-limit arXiv/Semantic Scholar calls: max 3 req/s

---

## 8. Implementation Order

Implement in exactly this order. Do not start a phase until the previous
phase's Definition of Done is fully satisfied.

```
Phase 0  — project setup, dependency additions, exception hierarchy, test fixtures
Phase 1  — ingestion (pdf_parser, paper_fetcher, CLI command, tests)
Phase 2  — understanding (models, agent, graph builder, CLI command, tests)
Phase 3  — planning (models, planner, validation, CLI command, tests)
Phase 4  — generation (module_agent, orchestrator, CLI command, tests)
Phase 5  — execution (sandbox.Dockerfile, sandbox runner, healer, scorer, tests)
Phase 6  — product (scaffolder, all templates, CLI command, tests)
Phase 7  — knowledge base (store, workspace, integration into Phases 4+5)
Phase 8  — from-paper unified command (wire all phases)
Phase 9  — TUI extension (new screens, progress streaming)
Phase 10 — polish (error messages, edge cases, performance profiling)
```

---

## 9. Definitions of Done

### Phase 0
- [ ] `scholardevclaw/exceptions.py` exists with full hierarchy
- [ ] All dependencies added to `pyproject.toml` under correct extras
- [ ] `core/tests/fixtures/` created with all fixture files
- [ ] Pre-commit hooks installed and passing on existing codebase
- [ ] `docker build -f docker/sandbox.Dockerfile -t sdc-sandbox:latest .` completes

### Phase 1 (Ingestion)
- [ ] `scholardevclaw ingest arxiv:1706.03762` downloads and parses Attention Is All You Need
- [ ] Resulting `paper_document.json` has `algorithms` length ≥ 2, `equations` length ≥ 10, `domain == "nlp"`
- [ ] `scholardevclaw ingest /path/to/any_local_paper.pdf` works
- [ ] `PaperDocument.to_dict()` and `from_dict()` roundtrip without loss
- [ ] All tests in `test_ingestion_pdf_parser.py` and `test_ingestion_paper_fetcher.py` pass
- [ ] Caching works: second call to same arXiv ID reads from cache, makes zero network calls

### Phase 2 (Understanding)
- [ ] `scholardevclaw understand paper_document.json` on Attention paper produces understanding where:
  - `complexity` is `"medium"`
  - `requirements` includes a PyTorch or equivalent library entry
  - `core_algorithm_description` mentions multi-head attention, positional encoding, encoder, decoder
  - `concept_nodes` length ≥ 6
  - `concept_edges` length ≥ 5
  - `confidence` ≥ 0.7
- [ ] `concept_graph.json` is valid networkx node_link_data format
- [ ] Understanding agent handles papers > 100k tokens without crashing (truncation works)
- [ ] All tests pass with mocked Anthropic client

### Phase 3 (Planning)
- [ ] `scholardevclaw plan understanding.json` on Attention understanding produces plan with ≥ 7 modules
- [ ] Plan includes: `data_loader`, `positional_encoding`, `multi_head_attention`, `transformer_model`,
  `train_loop`, `evaluate`, minimum
- [ ] `networkx.is_dag(plan_graph)` returns `True` (no circular dependencies)
- [ ] All `depends_on` references are valid module ids in the same plan
- [ ] Tech stack selector returns `numpy-only` for a systems-domain understanding with no ML libs
- [ ] All tests pass

### Phase 4 (Generation)
- [ ] `scholardevclaw generate plan.json understanding.json` produces all planned files
- [ ] All generated `.py` files parse with `ast.parse()` — zero syntax errors in final output
- [ ] `generation_report.json` contains `success_rate`, `duration_seconds`, per-module attempt counts
- [ ] Parallel generation with `--max-parallel 4` is at least 2× faster than `--max-parallel 1`
  on a ≥ 5 module plan (assert in integration test)
- [ ] Each generated module has a corresponding test file written alongside it
- [ ] Knowledge base context injection is visible in generation logs at DEBUG level

### Phase 5 (Execution)
- [ ] `SandboxRunner.run_tests(passing_project_fixture)` returns `ExecutionReport` with `success=True`
- [ ] `SandboxRunner.run_tests(broken_project_fixture)` returns `success=False` with non-empty `stderr`
- [ ] `SelfHealingLoop.heal()` reduces `tests_failed` on a seeded-broken generation
- [ ] Container has no internet access (verify: `docker inspect` shows `NetworkMode: none`)
- [ ] Memory limit is enforced (verify: process killed at limit, not hanging)
- [ ] `ReproducibilityScorer` returns `score >= 0.9` when stdout contains metrics ≥ 90% of claimed values

### Phase 6 (Product)
- [ ] `scholardevclaw scaffold project_dir` produces all six artifacts: `api/main.py`, `demo.py`,
  `pyproject.toml`, `Dockerfile`, `README.md`, `.github/workflows/ci.yml`
- [ ] `uvicorn api.main:app` starts without import errors
- [ ] `python demo.py --help` exits code 0
- [ ] Generated README has non-empty reproducibility table with at least one metric row
- [ ] Generated `pyproject.toml` is valid TOML that `pip install -e .` accepts

### Phase 7 (Knowledge Base)
- [ ] After running full pipeline on 2 papers, `scholardevclaw kb stats` shows 2 papers, ≥ 10 implementations
- [ ] `scholardevclaw kb search "layer normalization"` returns the correct paper if ingested
- [ ] KB persists across process restarts
- [ ] Third paper in same domain shows lower median `generation_attempts` vs first paper (KB helping)

### Phase 8 (from-paper)
- [ ] `scholardevclaw from-paper arxiv:1706.03762 --dry-run` completes in < 30s, shows correct plan
- [ ] `scholardevclaw from-paper arxiv:1706.03762 --heal --scaffold` runs full pipeline,
  `success_rate >= 0.7`, at least one test passes, README generated
- [ ] `--resume SESSION_ID` resumes a session that was killed at phase 4 (generation)
- [ ] Progress output matches the exact format specified in section 5 above

### Phase 9 (TUI)
- [ ] All 6 new screens launch without crashing
- [ ] `PlanningScreen` shows approval gate — generation does NOT start until approved
- [ ] `GenerationScreen` updates in real-time as modules complete
- [ ] Existing TUI screens are unchanged and still work

---

## 10. What "Claude Code for Research" Actually Means

You are building the tool that the best AI research labs in the world would
want to build but haven't because it requires solving too many subproblems
at once. Here is what that bar requires:

**It must be fast.** A researcher who finishes reading a paper should have
a running prototype by the time they're done with their coffee. That means
the full pipeline on a medium-complexity paper must complete in under 5 minutes.
Achieve this through aggressive parallelism and smart context windowing.

**It must be honest.** Never claim the implementation is correct when it
isn't. Show the reproducibility score prominently. Show what metrics
weren't reached. Researchers will distrust and abandon any tool that
oversells its outputs.

**It must compound.** Every paper implemented makes the next one faster
and better. The knowledge base is not a nice-to-have — it is the moat.
After 100 papers, ScholarDevClaw should be dramatically better than after 10.

**It must handle the full research stack.** Not just transformers and PyTorch.
Biology papers with protein sequences. Systems papers with C-like pseudocode.
RL papers with environment code. Multi-modal papers with vision and language
components. The architecture must be domain-agnostic from the start.

**It must be composable.** Every layer is independently useful. A researcher
who just wants to extract and understand a paper's algorithm without generating
code should be able to. A researcher who already has code and just wants
the product scaffold should be able to. Every layer has a clean CLI interface.

**It must be trustworthy.** Protected-branch blocking. Approval gates.
Confidence scores on every output. Sandbox isolation. Reproducibility
verification. Security researchers and ML researchers alike must be able
to trust what this tool does to their environment.

The test of whether you have built this correctly:
a first-year PhD student in ML, who has never heard of ScholarDevClaw,
reads the README, installs it, runs `from-paper` on the paper they're
currently reading, and 10 minutes later has a running implementation
they are not embarrassed to share with their advisor.

Build that tool. Nothing less.

---

*End of master agent prompt.*
*Session starts now. Read. Then build Phase 0.*