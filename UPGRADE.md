# UPGRADE.md — ScholarDevClaw v3: Paper to Product

> **Agent instructions:** Read this entire file before writing a single line of code.
> Implement phases in strict order. Do not skip ahead. Each phase has a
> "Definition of Done" — only advance when every item in it passes.
> When in doubt about naming, location, or interface: follow what already
> exists in `core/` rather than inventing something new.

---

## North Star

Transform ScholarDevClaw from a *codebase improvement suggester* into a
*research-to-product pipeline*. The end state: a researcher drops a PDF (or
DOI, or arXiv ID) and receives a fully working, benchmarked, packaged
implementation — zero existing codebase required.

The six-layer pipeline is:

```
[Ingestion] → [Understanding] → [Planning] → [Generation] → [Execution] → [Product]
                                      ↕
                              [Knowledge Base]   ← cross-cuts every layer
```

---

## Repo map (existing, do not move)

```
ScholarDevClaw/
├── core/                  # Python CLI/TUI — primary implementation target
│   ├── scholardevclaw/    # main package
│   │   ├── cli.py
│   │   ├── analyzer.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── specs/
│   └── pyproject.toml
├── agent/                 # agent orchestration layer
├── convex/                # backend state (Convex)
├── web/                   # TypeScript frontend
├── docker/                # Docker compose configs
├── scripts/               # runbook.sh and helpers
└── docs/
```

All new Python code lives inside `core/scholardevclaw/` unless stated
otherwise. All new modules are importable as
`from scholardevclaw.<module> import <Class>`.

---

## Dependency additions (add to `core/pyproject.toml`)

```toml
[project.optional-dependencies]
ingestion = [
    "pymupdf>=1.24.0",         # PDF parsing (import as fitz)
    "pdfplumber>=0.11.0",      # table + layout extraction
    "arxiv>=2.1.0",            # arXiv API client
    "semanticscholar>=0.8.0",  # Semantic Scholar API
    "pillow>=10.0.0",          # figure extraction
    "camelot-py[cv]>=0.11.0",  # table extraction from PDFs
]
understanding = [
    "anthropic>=0.25.0",       # primary LLM calls
    "openai>=1.30.0",          # fallback LLM
    "tiktoken>=0.7.0",         # token counting
    "networkx>=3.3",           # concept graph
]
execution = [
    "docker>=7.0.0",           # Docker SDK for sandboxed runs
    "psutil>=5.9.0",           # resource monitoring
    "pytest>=8.0.0",           # test runner
    "coverage>=7.5.0",
]
knowledge = [
    "chromadb>=0.5.0",         # local vector store (zero infra)
    "sentence-transformers>=3.0.0",  # embeddings
    "qdrant-client>=1.9.0",    # optional: production vector store
]
product = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "gradio>=4.36.0",
    "jinja2>=3.1.0",           # template rendering
    "tomlkit>=0.13.0",         # pyproject.toml generation
]

# mega-install for full pipeline
all = [
    "scholardevclaw[ingestion,understanding,execution,knowledge,product]"
]
```

---

## Phase 1 — PDF Ingestion Pipeline

**Goal:** Accept a PDF file path, DOI, or arXiv ID and produce a structured
`PaperDocument` object that all downstream layers consume.

### 1.1 Create `core/scholardevclaw/ingestion/__init__.py`

Empty init, re-exports `PaperDocument`, `PaperIngester`.

### 1.2 Create `core/scholardevclaw/ingestion/models.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class Equation:
    latex: str          # raw LaTeX string
    description: str    # surrounding text context (≤200 chars)
    page: int

@dataclass
class Algorithm:
    name: str           # e.g. "Algorithm 1: Training Procedure"
    pseudocode: str     # verbatim extracted text
    page: int
    language_hint: str  # "python-like" | "math" | "unknown"

@dataclass
class Figure:
    caption: str
    page: int
    image_path: Optional[Path]  # saved PNG, None if extraction failed

@dataclass
class Section:
    title: str
    level: int          # 1=top, 2=sub, 3=subsub
    content: str        # full text of section
    page_start: int

@dataclass
class PaperDocument:
    # identifiers
    title: str
    authors: list[str]
    arxiv_id: Optional[str]
    doi: Optional[str]
    year: Optional[int]
    abstract: str

    # structured content
    sections: list[Section]
    equations: list[Equation]
    algorithms: list[Algorithm]
    figures: list[Figure]

    # raw
    full_text: str
    pdf_path: Optional[Path]

    # metadata
    references: list[str]       # raw reference strings
    keywords: list[str]
    domain: str = "unknown"     # "cv" | "nlp" | "rl" | "systems" | "theory"
```

### 1.3 Create `core/scholardevclaw/ingestion/pdf_parser.py`

Implement class `PDFParser` with these methods. Use `pymupdf` (fitz) as
primary, `pdfplumber` as fallback for tables:

```python
class PDFParser:
    def parse(self, pdf_path: Path) -> PaperDocument: ...

    def _extract_text_by_section(self, doc) -> list[Section]: ...
    # heuristic: lines in larger font weight = section headers
    # use fitz block analysis, sort by font size descending

    def _extract_equations(self, doc) -> list[Equation]: ...
    # look for LaTeX delimiters in text: $...$, \begin{equation}...\end{equation}
    # also look for lines that are purely symbolic (high ratio of non-alpha chars)

    def _extract_algorithms(self, doc) -> list[Algorithm]: ...
    # target blocks with "Algorithm N" header pattern
    # extract verbatim until next section-level header

    def _extract_figures(self, doc, output_dir: Path) -> list[Figure]: ...
    # use fitz.get_images(), save as PNG to output_dir/figures/

    def _detect_domain(self, text: str) -> str: ...
    # keyword matching: {"transformer","bert","gpt","attention"} → "nlp"
    # {"convolution","resnet","yolo","segmentation"} → "cv"
    # {"reward","policy","q-learning","environment"} → "rl"
    # {"kernel","mutex","scheduler","memory"} → "systems"
    # default → "theory"
```

### 1.4 Create `core/scholardevclaw/ingestion/paper_fetcher.py`

```python
class PaperFetcher:
    """Resolve DOI / arXiv ID / URL → downloaded PDF + metadata."""

    def fetch_by_arxiv_id(self, arxiv_id: str, dest_dir: Path) -> PaperDocument:
        # use arxiv library: arxiv.Search(id_list=[arxiv_id])
        # download PDF to dest_dir/<arxiv_id>.pdf
        # parse with PDFParser
        # merge arxiv metadata (authors, abstract, year) into PaperDocument
        ...

    def fetch_by_doi(self, doi: str, dest_dir: Path) -> PaperDocument:
        # call Semantic Scholar API: https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>
        # fields=title,authors,year,abstract,openAccessPdf
        # if openAccessPdf exists, download and parse
        # else raise PaperNotAccessibleError with helpful message
        ...

    def fetch_by_url(self, url: str, dest_dir: Path) -> PaperDocument:
        # HEAD request to detect content-type
        # if application/pdf → download + parse
        # if html → look for meta[name=citation_pdf_url] or .pdf link
        ...
```

### 1.5 Wire into CLI

In `core/scholardevclaw/cli.py`, add command:

```
scholardevclaw ingest <pdf_or_doi_or_url> [--output-dir DIR]
```

Output: JSON-serialized `PaperDocument` saved to
`<output-dir>/paper_document.json`.

### 1.6 Definition of Done — Phase 1

- [ ] `scholardevclaw ingest arxiv:1706.03762` downloads Attention Is All You
  Need and produces `paper_document.json` with `algorithms` list non-empty
  and `equations` list length > 10.
- [ ] `scholardevclaw ingest /path/to/local.pdf` works for any ML paper.
- [ ] `PaperDocument` serializes to/from JSON without loss (`to_dict()` /
  `from_dict()` methods present on model).
- [ ] Unit tests in `core/tests/test_ingestion.py` pass for PDF parser and
  fetcher (use `arxiv:2005.14165` — GPT-3 paper — as integration test fixture).

---

## Phase 2 — Paper Understanding Agent

**Goal:** Feed a `PaperDocument` to an LLM agent and produce a structured
`PaperUnderstanding` object: the agent's answer to "what does this paper
actually do, and what do I need to implement it?"

### 2.1 Create `core/scholardevclaw/understanding/__init__.py`

Re-exports `PaperUnderstanding`, `UnderstandingAgent`.

### 2.2 Create `core/scholardevclaw/understanding/models.py`

```python
@dataclass
class Contribution:
    claim: str           # one sentence
    novelty: str         # what is new vs prior work
    is_implementable: bool  # can this be coded up?

@dataclass
class Requirement:
    name: str            # e.g. "ImageNet dataset"
    type: str            # "dataset" | "library" | "hardware" | "baseline"
    is_optional: bool
    notes: str

@dataclass
class ConceptNode:
    id: str
    label: str
    type: str            # "model" | "operation" | "loss" | "dataset" | "metric"
    description: str

@dataclass
class ConceptEdge:
    source_id: str
    target_id: str
    relation: str        # "uses" | "produces" | "compared_against" | "trained_on"

@dataclass
class PaperUnderstanding:
    paper_title: str
    one_line_summary: str      # ≤ 20 words, what the paper does
    problem_statement: str     # what problem this solves
    key_insight: str           # the core idea in 2-3 sentences

    contributions: list[Contribution]
    requirements: list[Requirement]

    concept_nodes: list[ConceptNode]
    concept_edges: list[ConceptEdge]

    # what a developer needs to know
    core_algorithm_description: str   # plain English, no jargon
    input_output_spec: str            # "takes X, produces Y"
    evaluation_protocol: str          # how the paper measures success

    complexity: str                   # "low" | "medium" | "high" | "research-only"
    estimated_impl_hours: int         # rough estimate
    confidence: float                 # 0.0–1.0, agent's confidence in its read
```

### 2.3 Create `core/scholardevclaw/understanding/agent.py`

This is the most critical file in the entire upgrade. Implement with care.

```python
import anthropic
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.understanding.models import PaperUnderstanding
import json

SYSTEM_PROMPT = """You are an expert ML researcher and senior software engineer.
You read research papers with precision and extract structured information
that allows a developer to implement the paper from scratch.
Always respond with valid JSON matching the exact schema provided.
Never hallucinate citations or results. If unsure, say so in the confidence field.
"""

class UnderstandingAgent:
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def understand(self, doc: PaperDocument) -> PaperUnderstanding:
        prompt = self._build_prompt(doc)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        data = self._parse_json_response(raw)
        return PaperUnderstanding(**data)

    def _build_prompt(self, doc: PaperDocument) -> str:
        # Build a focused prompt — do NOT dump the entire paper.
        # Include: title, abstract, algorithm blocks, key equations, conclusion section.
        # Stay under 80k tokens total.
        algo_text = "\n\n".join(
            f"=== {a.name} ===\n{a.pseudocode}" for a in doc.algorithms
        )
        eq_text = "\n".join(
            f"[Eq on p.{e.page}]: {e.latex}  ({e.description})"
            for e in doc.algorithms[:20]   # cap at 20 equations
        )
        conclusion = next(
            (s.content for s in doc.sections if "conclusion" in s.title.lower()),
            ""
        )
        return f"""Paper: {doc.title}
Authors: {', '.join(doc.authors)}
Abstract: {doc.abstract}

Algorithms:
{algo_text}

Key Equations:
{eq_text}

Conclusion:
{conclusion[:3000]}

---
Analyze this paper and return a JSON object with exactly these fields:
{{
  "paper_title": str,
  "one_line_summary": str,   // ≤20 words
  "problem_statement": str,
  "key_insight": str,
  "contributions": [
    {{"claim": str, "novelty": str, "is_implementable": bool}}
  ],
  "requirements": [
    {{"name": str, "type": "dataset|library|hardware|baseline", "is_optional": bool, "notes": str}}
  ],
  "concept_nodes": [
    {{"id": str, "label": str, "type": "model|operation|loss|dataset|metric", "description": str}}
  ],
  "concept_edges": [
    {{"source_id": str, "target_id": str, "relation": "uses|produces|compared_against|trained_on"}}
  ],
  "core_algorithm_description": str,
  "input_output_spec": str,
  "evaluation_protocol": str,
  "complexity": "low|medium|high|research-only",
  "estimated_impl_hours": int,
  "confidence": float   // 0.0 to 1.0
}}
Return only the JSON object. No markdown fences. No preamble.
"""

    def _parse_json_response(self, raw: str) -> dict:
        # strip any accidental markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
```

### 2.4 Build concept graph from `PaperUnderstanding`

In `core/scholardevclaw/understanding/graph.py`:

```python
import networkx as nx
from scholardevclaw.understanding.models import PaperUnderstanding

def build_concept_graph(understanding: PaperUnderstanding) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in understanding.concept_nodes:
        G.add_node(node.id, label=node.label, type=node.type,
                   description=node.description)
    for edge in understanding.concept_edges:
        G.add_edge(edge.source_id, edge.target_id, relation=edge.relation)
    return G

def export_graph_json(G: nx.DiGraph) -> dict:
    return nx.node_link_data(G)
```

### 2.5 Wire into CLI

```
scholardevclaw understand <paper_document.json> [--model MODEL] [--output-dir DIR]
```

Output: `understanding.json` + `concept_graph.json`.

### 2.6 Definition of Done — Phase 2

- [ ] `scholardevclaw understand paper_document.json` on the Attention Is All
  You Need document produces an `understanding.json` where
  `complexity == "medium"`, `requirements` includes PyTorch, and
  `core_algorithm_description` mentions multi-head attention and positional
  encoding without reading like a copy-paste of the abstract.
- [ ] `concept_graph.json` has at least 6 nodes and 5 edges for any
  Transformer-family paper.
- [ ] Agent gracefully handles token-limit scenarios by truncating
  `algo_text` and `eq_text` while keeping abstract and conclusion intact.
- [ ] Unit tests in `core/tests/test_understanding.py` mock the Anthropic
  client and verify JSON parsing + dataclass construction for a fixture
  response.

---

## Phase 3 — Dynamic Implementation Planner

**Goal:** Replace the static hand-written `specs/` system with a dynamic
planner that derives a module breakdown from `PaperUnderstanding`.

### 3.1 Create `core/scholardevclaw/planning/__init__.py`

Re-exports `ImplementationPlan`, `ImplementationPlanner`.

### 3.2 Create `core/scholardevclaw/planning/models.py`

```python
@dataclass
class CodeModule:
    id: str                      # e.g. "data_loader", "model_backbone"
    name: str                    # human-readable
    description: str             # what this module does
    file_path: str               # relative path in generated project
    depends_on: list[str]        # other module ids this imports from
    priority: int                # generation order (lower = first)
    estimated_lines: int
    test_file_path: str          # corresponding test file
    tech_stack: str              # "pytorch" | "jax" | "numpy" | "stdlib" | "fastapi"

@dataclass
class ImplementationPlan:
    project_name: str            # snake_case, derived from paper title
    target_language: str         # always "python" for now
    tech_stack: str              # "pytorch" | "jax" | "numpy-only"
    modules: list[CodeModule]    # ordered by priority ascending
    directory_structure: dict    # tree dict representing file layout
    environment: dict            # {package: version} for requirements.txt
    entry_points: list[str]      # module ids that are CLI/API entry points
    estimated_total_lines: int
```

### 3.3 Create `core/scholardevclaw/planning/planner.py`

```python
class ImplementationPlanner:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def plan(
        self,
        understanding: PaperUnderstanding,
        doc: PaperDocument,
    ) -> ImplementationPlan:
        stack = self._select_tech_stack(understanding, doc)
        plan_data = self._llm_plan(understanding, stack)
        return ImplementationPlan(**plan_data)

    def _select_tech_stack(
        self, understanding: PaperUnderstanding, doc: PaperDocument
    ) -> str:
        """Rule-based stack selection. No LLM needed here."""
        domain = doc.domain
        reqs = {r.name.lower() for r in understanding.requirements}
        text = understanding.core_algorithm_description.lower()
        # JAX for papers that explicitly mention it or TPU training
        if "jax" in reqs or "tpu" in text:
            return "jax"
        # Systems papers → stdlib + numpy
        if domain == "systems" and "pytorch" not in reqs:
            return "numpy-only"
        # Everything else → PyTorch (safe default for ML)
        return "pytorch"

    def _llm_plan(self, understanding: PaperUnderstanding, stack: str) -> dict:
        prompt = f"""You are a senior ML engineer planning the implementation of a research paper.

Paper: {understanding.paper_title}
Summary: {understanding.one_line_summary}
Core algorithm: {understanding.core_algorithm_description}
Input/output: {understanding.input_output_spec}
Tech stack: {stack}
Requirements: {[r.name for r in understanding.requirements if not r.is_optional]}
Evaluation: {understanding.evaluation_protocol}
Complexity: {understanding.complexity}

Design a complete, production-quality Python project structure to implement this paper.
Return a JSON object with exactly these fields:
{{
  "project_name": str,             // snake_case
  "target_language": "python",
  "tech_stack": "{stack}",
  "modules": [
    {{
      "id": str,                   // snake_case unique identifier
      "name": str,
      "description": str,
      "file_path": str,            // e.g. "src/model/attention.py"
      "depends_on": [str],         // list of other module ids
      "priority": int,             // 1=first to implement
      "estimated_lines": int,
      "test_file_path": str,       // e.g. "tests/test_attention.py"
      "tech_stack": str
    }}
  ],
  "directory_structure": {{
    "src/": {{
      "model/": {{}},
      "data/": {{}}
    }},
    "tests/": {{}},
    "scripts/": {{}}
  }},
  "environment": {{"torch": ">=2.0.0", "numpy": ">=1.26.0"}},
  "entry_points": [str],
  "estimated_total_lines": int
}}

Rules:
- Every paper needs at minimum: a data loader, a model definition, a training loop,
  an evaluation script, and a README generator.
- tests/ must mirror src/ structure exactly.
- Modules must be ordered so that a module's dependencies are always lower priority numbers.
- Return only JSON. No markdown. No preamble.
"""
        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
```

### 3.4 Backward compatibility

Keep existing `specs/` directory intact. Add a `--use-specs` flag to legacy
commands. New `from-paper` workflow always uses the dynamic planner.

### 3.5 Wire into CLI

```
scholardevclaw plan <understanding.json> [--stack pytorch|jax|numpy] [--output-dir DIR]
```

Output: `implementation_plan.json`.

### 3.6 Definition of Done — Phase 3

- [ ] Running `plan` on the Attention Is All You Need understanding produces
  a plan with ≥ 7 modules including at minimum: `data_loader`,
  `positional_encoding`, `multi_head_attention`, `transformer_model`,
  `train_loop`, `evaluate`, `readme_generator`.
- [ ] Tech stack selector chooses `numpy-only` for a systems paper that has
  no deep learning requirements.
- [ ] Module priorities form a valid topological order (no module has a
  dependency with higher priority number).
- [ ] Unit test verifies topological ordering constraint on mock plan data.

---

## Phase 4 — Multi-Agent Code Generator

**Goal:** Take an `ImplementationPlan` and generate every module as working,
documented, typed Python code. Use parallel async sub-agents with an
orchestrator. Include a self-healing loop for import and type errors.

### 4.1 Create `core/scholardevclaw/generation/__init__.py`

Re-exports `CodeOrchestrator`, `GenerationResult`.

### 4.2 Create `core/scholardevclaw/generation/models.py`

```python
@dataclass
class ModuleResult:
    module_id: str
    file_path: str
    code: str
    test_code: str
    generation_attempts: int
    final_errors: list[str]    # empty = success
    tokens_used: int

@dataclass
class GenerationResult:
    plan: ImplementationPlan
    module_results: list[ModuleResult]
    output_dir: Path
    success_rate: float        # modules generated without errors / total
    total_tokens_used: int
    duration_seconds: float
```

### 4.3 Create `core/scholardevclaw/generation/module_agent.py`

```python
import asyncio
import anthropic

MODULE_SYSTEM_PROMPT = """You are an expert Python developer implementing a
research paper. Write production-quality, fully typed Python code.
Rules:
- Use type hints everywhere (Python 3.11+).
- Every public function has a docstring.
- Imports are at the top, stdlib → third-party → local.
- No placeholder comments like "# TODO: implement". Write real code.
- Match the tech stack specified. If PyTorch: use nn.Module, DataLoader etc.
- Code must be importable without errors.
- Return ONLY the Python code. No markdown fences. No explanation.
"""

TEST_SYSTEM_PROMPT = """You are an expert in pytest. Write comprehensive tests
for the given Python module. Use pytest fixtures. Mock external dependencies.
Include at least one happy-path test and one edge-case test per public function.
Return ONLY the pytest code. No markdown. No explanation.
"""

class ModuleAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str):
        self.client = client
        self.model = model

    async def generate(
        self,
        module: CodeModule,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        context_modules: dict[str, str],  # id → already-generated code
        max_attempts: int = 3,
    ) -> ModuleResult:
        attempts = 0
        errors = []
        code = ""
        test_code = ""

        while attempts < max_attempts:
            attempts += 1
            prompt = self._build_module_prompt(
                module, plan, understanding, context_modules, errors
            )
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=MODULE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            code = response.content[0].text.strip()

            # syntax check
            syntax_errors = self._check_syntax(code)
            if not syntax_errors:
                break
            errors = syntax_errors

        # generate tests for the final code
        test_code = await self._generate_tests(module, code, understanding)

        return ModuleResult(
            module_id=module.id,
            file_path=module.file_path,
            code=code,
            test_code=test_code,
            generation_attempts=attempts,
            final_errors=errors,
            tokens_used=0,  # sum from responses if needed
        )

    def _build_module_prompt(
        self,
        module: CodeModule,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        context_modules: dict[str, str],
        prior_errors: list[str],
    ) -> str:
        context_block = ""
        for dep_id in module.depends_on:
            if dep_id in context_modules:
                context_block += f"\n# Already implemented: {dep_id}\n"
                context_block += context_modules[dep_id][:2000]  # cap per dep

        error_block = ""
        if prior_errors:
            error_block = f"\nPrevious attempt had these errors — fix them:\n"
            error_block += "\n".join(prior_errors)

        return f"""Paper: {understanding.paper_title}
Core algorithm: {understanding.core_algorithm_description}
Tech stack: {plan.tech_stack}

Implement this module:
Module: {module.name} ({module.id})
File: {module.file_path}
Description: {module.description}
Estimated lines: {module.estimated_lines}
{error_block}

Context from dependencies:
{context_block}

Write complete, production-quality Python code for {module.file_path}.
"""

    def _check_syntax(self, code: str) -> list[str]:
        import ast
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    async def _generate_tests(
        self, module: CodeModule, code: str, understanding: PaperUnderstanding
    ) -> str:
        prompt = f"""Module to test: {module.name} ({module.file_path})
Paper context: {understanding.one_line_summary}
Code:
{code[:6000]}

Write pytest tests for this module."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=TEST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
```

### 4.4 Create `core/scholardevclaw/generation/orchestrator.py`

```python
import asyncio
import time
from pathlib import Path
import anthropic

class CodeOrchestrator:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate_all(
        self,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        output_dir: Path,
        max_parallel: int = 4,
    ) -> GenerationResult:
        start = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        # topological sort by priority
        sorted_modules = sorted(plan.modules, key=lambda m: m.priority)

        results: dict[str, ModuleResult] = {}
        context_modules: dict[str, str] = {}  # id → code (for dependency context)

        # group modules by priority level for parallel execution
        from itertools import groupby
        priority_groups = groupby(sorted_modules, key=lambda m: m.priority)

        for priority, group in priority_groups:
            batch = list(group)
            agent = ModuleAgent(self.client, self.model)
            tasks = [
                agent.generate(m, plan, understanding, context_modules)
                for m in batch
            ]
            # run current priority group in parallel, capped at max_parallel
            semaphore = asyncio.Semaphore(max_parallel)
            async def bounded(task):
                async with semaphore:
                    return await task
            batch_results = await asyncio.gather(*[bounded(t) for t in tasks])

            for result in batch_results:
                results[result.module_id] = result
                if not result.final_errors:
                    context_modules[result.module_id] = result.code
                # write files
                self._write_module(result, output_dir)

        success_count = sum(1 for r in results.values() if not r.final_errors)
        return GenerationResult(
            plan=plan,
            module_results=list(results.values()),
            output_dir=output_dir,
            success_rate=success_count / len(results) if results else 0.0,
            total_tokens_used=sum(r.tokens_used for r in results.values()),
            duration_seconds=time.time() - start,
        )

    def _write_module(self, result: ModuleResult, output_dir: Path) -> None:
        target = output_dir / result.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.code)

        test_target = output_dir / result.test_file_path  # from module definition
        test_target.parent.mkdir(parents=True, exist_ok=True)
        test_target.write_text(result.test_code)

    def generate_sync(self, plan, understanding, output_dir, max_parallel=4):
        """Synchronous wrapper for CLI use."""
        return asyncio.run(
            self.generate_all(plan, understanding, output_dir, max_parallel)
        )
```

### 4.5 Wire into CLI

```
scholardevclaw generate <implementation_plan.json> <understanding.json> \
    [--output-dir DIR] [--max-parallel N] [--model MODEL]
```

Output: full project directory tree + `generation_report.json`.

### 4.6 Definition of Done — Phase 4

- [ ] `generate` on a simple paper (e.g. Word2Vec — arxiv not needed, write
  an `understanding.json` fixture by hand) produces all planned files in
  correct directory structure.
- [ ] Every generated `.py` file passes `ast.parse()` — zero syntax errors
  allowed in final output.
- [ ] `generation_report.json` contains `success_rate`, `duration_seconds`,
  per-module attempt counts.
- [ ] Running with `--max-parallel 4` is demonstrably faster than `--max-parallel 1`
  on a ≥ 5-module plan (add a timing assertion to the integration test).
- [ ] `generation/tests/test_orchestrator.py` mocks `anthropic.AsyncAnthropic`
  and verifies parallel execution and file writing.

---

## Phase 5 — Sandboxed Execution + Self-Healing Loop

**Goal:** Run the generated code in an isolated Docker container, capture
all output and errors, compute a reproducibility score by comparing to
claimed paper metrics, and feed failures back to the generator.

### 5.1 Create `core/scholardevclaw/execution/__init__.py`

Re-exports `SandboxRunner`, `ExecutionReport`, `ReproducibilityScorer`.

### 5.2 Create `core/scholardevclaw/execution/sandbox.py`

```python
import docker
import tarfile
import io
import json
from pathlib import Path
from dataclasses import dataclass

SANDBOX_IMAGE = "scholardevclaw-sandbox:latest"
# Build once from docker/sandbox.Dockerfile (created below)

@dataclass
class ExecutionReport:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    peak_memory_mb: float
    tests_passed: int
    tests_failed: int
    tests_errors: int
    success: bool   # exit_code == 0 and tests_failed == 0

class SandboxRunner:
    def __init__(self, timeout_seconds: int = 300, memory_limit_mb: int = 4096):
        self.client = docker.from_env()
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit_mb

    def run_tests(self, project_dir: Path) -> ExecutionReport:
        """Run pytest inside a container against the generated project."""
        container = self.client.containers.run(
            image=SANDBOX_IMAGE,
            command="pytest tests/ -v --json-report --json-report-file=/tmp/report.json",
            volumes={str(project_dir): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            mem_limit=f"{self.memory_limit}m",
            network_disabled=True,         # no network access in sandbox
            remove=False,                  # keep to extract report
            detach=True,
        )
        try:
            result = container.wait(timeout=self.timeout)
            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()
            exit_code = result["StatusCode"]

            # extract pytest JSON report
            report = self._extract_json_report(container)
            passed = report.get("summary", {}).get("passed", 0)
            failed = report.get("summary", {}).get("failed", 0)
            errors = report.get("summary", {}).get("errors", 0)

            return ExecutionReport(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=0,  # parse from report if needed
                peak_memory_mb=0,
                tests_passed=passed,
                tests_failed=failed,
                tests_errors=errors,
                success=(exit_code == 0 and failed == 0),
            )
        finally:
            container.remove(force=True)

    def _extract_json_report(self, container) -> dict:
        try:
            bits, _ = container.get_archive("/tmp/report.json")
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tar:
                f = tar.extractfile("report.json")
                return json.load(f)
        except Exception:
            return {}
```

### 5.3 Create `docker/sandbox.Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pytest \
    pytest-json-report \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    jax[cpu] \
    transformers \
    datasets \
    scikit-learn \
    fastapi \
    uvicorn \
    gradio

WORKDIR /workspace
CMD ["pytest", "tests/", "-v"]
```

Build this image once during `scripts/runbook.sh dev setup`:

```bash
docker build -f docker/sandbox.Dockerfile -t scholardevclaw-sandbox:latest .
```

### 5.4 Create `core/scholardevclaw/execution/scorer.py`

```python
@dataclass
class ReproducibilityReport:
    paper_title: str
    claimed_metrics: dict[str, float]   # from understanding
    achieved_metrics: dict[str, float]  # extracted from stdout
    delta: dict[str, float]             # achieved - claimed
    score: float                        # 0.0–1.0
    verdict: str                        # "reproduced" | "partial" | "failed"

class ReproducibilityScorer:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def score(
        self,
        understanding: PaperUnderstanding,
        execution_report: ExecutionReport,
    ) -> ReproducibilityReport:
        # use LLM to extract metrics from stdout
        # e.g. "Accuracy: 92.3%" → {"accuracy": 0.923}
        achieved = self._extract_metrics_from_output(execution_report.stdout)
        claimed = self._extract_claimed_metrics(understanding)
        delta = {k: achieved.get(k, 0) - claimed.get(k, 0) for k in claimed}
        score = self._compute_score(claimed, achieved)
        verdict = "reproduced" if score > 0.9 else "partial" if score > 0.5 else "failed"
        return ReproducibilityReport(
            paper_title=understanding.paper_title,
            claimed_metrics=claimed,
            achieved_metrics=achieved,
            delta=delta,
            score=score,
            verdict=verdict,
        )

    def _extract_metrics_from_output(self, stdout: str) -> dict[str, float]:
        # prompt LLM: "Extract metric names and values from this output"
        # return {"metric_name": value}
        ...

    def _extract_claimed_metrics(self, understanding: PaperUnderstanding) -> dict[str, float]:
        # prompt LLM: "What metrics does this paper claim? Return {name: value}"
        ...

    def _compute_score(self, claimed: dict, achieved: dict) -> float:
        if not claimed:
            return 0.5  # can't verify
        scores = []
        for k, v in claimed.items():
            a = achieved.get(k, 0)
            if v == 0:
                continue
            ratio = min(a / v, v / a) if a > 0 else 0
            scores.append(ratio)
        return sum(scores) / len(scores) if scores else 0.0
```

### 5.5 Self-healing loop in `core/scholardevclaw/execution/healer.py`

```python
class SelfHealingLoop:
    """Runs execution, extracts errors, re-invokes generator on failed modules."""

    def __init__(self, orchestrator: CodeOrchestrator, runner: SandboxRunner,
                 max_healing_rounds: int = 3):
        self.orchestrator = orchestrator
        self.runner = runner
        self.max_rounds = max_healing_rounds

    def heal(
        self,
        generation_result: GenerationResult,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
    ) -> GenerationResult:
        current = generation_result
        for round_num in range(self.max_rounds):
            report = self.runner.run_tests(current.output_dir)
            if report.success:
                break
            # parse failures → identify failing modules
            failing_module_ids = self._identify_failing_modules(report)
            if not failing_module_ids:
                break
            # re-generate only failing modules, injecting error context
            current = self.orchestrator.generate_sync(
                plan=plan,
                understanding=understanding,
                output_dir=current.output_dir,
                module_filter=failing_module_ids,
                error_context=report.stderr,
            )
        return current

    def _identify_failing_modules(self, report: ExecutionReport) -> list[str]:
        # parse stderr for "FAILED tests/test_<module_id>.py"
        import re
        pattern = r"FAILED tests/test_(\w+)\.py"
        return re.findall(pattern, report.stderr)
```

### 5.6 Wire into CLI

```
scholardevclaw execute <project_dir> [--heal] [--timeout 300] [--output-dir DIR]
```

Output: `execution_report.json` + `reproducibility_report.json`.

### 5.7 Definition of Done — Phase 5

- [ ] `sandbox.Dockerfile` builds cleanly with `docker build`.
- [ ] `SandboxRunner.run_tests()` returns `ExecutionReport` with correct
  `tests_passed`/`tests_failed` counts for a toy project (fixture: a
  directory with one passing and one failing test).
- [ ] `SelfHealingLoop` reduces `tests_failed` on a seeded-broken generation
  (inject a known syntax error into a fixture module, verify healer removes it).
- [ ] `--heal` flag demonstrates a multi-round improvement in `generation_report.json`.
- [ ] Container has no network access (verify with `docker inspect`).

---

## Phase 6 — Product Scaffold Generator

**Goal:** Take a working, passing project directory and generate everything
needed to ship it: a FastAPI service, a Gradio demo, `pyproject.toml`,
Dockerfile, and documentation.

### 6.1 Create `core/scholardevclaw/product/__init__.py`

Re-exports `ProductScaffolder`.

### 6.2 Create `core/scholardevclaw/product/scaffolder.py`

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

class ProductScaffolder:
    def __init__(self, templates_dir: Path = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(templates_dir)))

    def scaffold(
        self,
        project_dir: Path,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        reproducibility_report: ReproducibilityReport,
    ) -> None:
        self._generate_api(project_dir, plan, understanding)
        self._generate_gradio_demo(project_dir, plan, understanding)
        self._generate_pyproject(project_dir, plan, understanding)
        self._generate_dockerfile(project_dir, plan)
        self._generate_readme(project_dir, plan, understanding, reproducibility_report)
        self._generate_github_actions(project_dir)

    def _generate_api(self, project_dir, plan, understanding):
        # Render templates/api_main.py.j2 → project_dir/api/main.py
        # Template vars: project_name, entry_point_module, input_output_spec
        template = self.env.get_template("api_main.py.j2")
        out = template.render(
            project_name=plan.project_name,
            summary=understanding.one_line_summary,
            io_spec=understanding.input_output_spec,
            entry_module=plan.entry_points[0] if plan.entry_points else "main",
        )
        (project_dir / "api").mkdir(exist_ok=True)
        (project_dir / "api" / "main.py").write_text(out)

    def _generate_gradio_demo(self, project_dir, plan, understanding):
        template = self.env.get_template("gradio_demo.py.j2")
        out = template.render(
            project_name=plan.project_name,
            summary=understanding.one_line_summary,
            io_spec=understanding.input_output_spec,
        )
        (project_dir / "demo.py").write_text(out)

    def _generate_pyproject(self, project_dir, plan, understanding):
        import tomlkit
        doc = tomlkit.document()
        project = tomlkit.table()
        project.add("name", plan.project_name)
        project.add("version", "0.1.0")
        project.add("description", understanding.one_line_summary)
        project.add("requires-python", ">=3.11")
        deps = [f"{k}{v}" for k, v in plan.environment.items()]
        project.add("dependencies", deps)
        doc.add("project", project)
        (project_dir / "pyproject.toml").write_text(tomlkit.dumps(doc))

    def _generate_dockerfile(self, project_dir, plan):
        template = self.env.get_template("Dockerfile.j2")
        out = template.render(
            project_name=plan.project_name,
            stack=plan.tech_stack,
            deps=list(plan.environment.keys()),
        )
        (project_dir / "Dockerfile").write_text(out)

    def _generate_readme(self, project_dir, plan, understanding, repro):
        template = self.env.get_template("README.md.j2")
        out = template.render(
            project_name=plan.project_name,
            paper_title=understanding.paper_title,
            summary=understanding.one_line_summary,
            key_insight=understanding.key_insight,
            problem=understanding.problem_statement,
            io_spec=understanding.input_output_spec,
            evaluation=understanding.evaluation_protocol,
            repro_score=repro.score,
            repro_verdict=repro.verdict,
            claimed_metrics=repro.claimed_metrics,
            achieved_metrics=repro.achieved_metrics,
            requirements=[r.name for r in understanding.requirements],
            stack=plan.tech_stack,
        )
        (project_dir / "README.md").write_text(out)

    def _generate_github_actions(self, project_dir):
        # Write .github/workflows/ci.yml with: install + pytest + docker build
        ci = """name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
"""
        actions_dir = project_dir / ".github" / "workflows"
        actions_dir.mkdir(parents=True, exist_ok=True)
        (actions_dir / "ci.yml").write_text(ci)
```

### 6.3 Create Jinja2 templates in `core/scholardevclaw/product/templates/`

#### `api_main.py.j2`

```python
"""FastAPI service for {{ project_name }}
Auto-generated by ScholarDevClaw from: {{ summary }}
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="{{ project_name }}",
    description="{{ summary }}",
    version="0.1.0",
)

class InputPayload(BaseModel):
    """{{ io_spec }}"""
    data: dict

class OutputPayload(BaseModel):
    result: dict

@app.post("/predict", response_model=OutputPayload)
async def predict(payload: InputPayload) -> OutputPayload:
    from src.{{ entry_module }} import run
    result = run(payload.data)
    return OutputPayload(result=result)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "{{ project_name }}"}
```

#### `gradio_demo.py.j2`

```python
"""Gradio demo for {{ project_name }}
{{ summary }}
"""
import gradio as gr

def run_inference(input_text: str) -> str:
    from src.{{ entry_module }} import run
    result = run({"text": input_text})
    return str(result)

demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="{{ project_name }}",
    description="{{ summary }}",
)

if __name__ == "__main__":
    demo.launch()
```

#### `README.md.j2`

```markdown
# {{ project_name }}

> Implementation of **{{ paper_title }}**

{{ summary }}

## What this implements

**Problem:** {{ problem }}

**Key insight:** {{ key_insight }}

**Input / Output:** {{ io_spec }}

## Reproducibility

| Metric | Claimed | Achieved |
|--------|---------|---------|
{% for k, v in claimed_metrics.items() -%}
| {{ k }} | {{ v }} | {{ achieved_metrics.get(k, "—") }} |
{% endfor %}

Reproducibility score: **{{ "%.1f"|format(repro_score * 100) }}%** ({{ repro_verdict }})

## Quickstart

```bash
pip install -e ".[all]"
python demo.py                   # Gradio UI
uvicorn api.main:app --reload    # FastAPI
pytest tests/ -v                 # Tests
```

## Tech stack

{{ stack }} · Python 3.11+

## Requirements

{% for r in requirements %}- {{ r }}
{% endfor %}

## Evaluation

{{ evaluation }}

---
*Generated by [ScholarDevClaw](https://github.com/Ronak-IIITD/ScholarDevClaw)*
```

### 6.4 Wire into CLI

```
scholardevclaw scaffold <project_dir> <plan.json> <understanding.json> \
    <reproducibility_report.json> [--output-dir DIR]
```

### 6.5 Definition of Done — Phase 6

- [ ] `scaffold` on any passing project produces `api/main.py`, `demo.py`,
  `pyproject.toml`, `Dockerfile`, `README.md`, `.github/workflows/ci.yml`.
- [ ] Generated FastAPI app starts without import errors:
  `uvicorn api.main:app --host 0.0.0.0 --port 8000` exits with code 0 on
  health check.
- [ ] Generated README has non-empty reproducibility table.
- [ ] Generated Gradio demo imports without errors.

---

## Phase 7 — Knowledge Base (cross-cutting, implement after Phase 3)

**Goal:** Store all processed papers and their implementations in a vector
database. Use past implementations as context for new generation requests.
This is what makes ScholarDevClaw compound value over time.

### 7.1 Create `core/scholardevclaw/knowledge/__init__.py`

Re-exports `KnowledgeBase`.

### 7.2 Create `core/scholardevclaw/knowledge/store.py`

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 33M params, fast, good quality

class KnowledgeBase:
    def __init__(self, persist_dir: Path = Path.home() / ".scholardevclaw" / "kb"):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.papers = self.client.get_or_create_collection("papers")
        self.implementations = self.client.get_or_create_collection("implementations")
        self.patterns = self.client.get_or_create_collection("patterns")

    def store_paper(self, doc: PaperDocument, understanding: PaperUnderstanding) -> None:
        text = f"{doc.title}. {doc.abstract}. {understanding.key_insight}"
        embedding = self.embedder.encode([text])[0].tolist()
        meta = {
            "title": doc.title,
            "domain": doc.domain,
            "complexity": understanding.complexity,
            "arxiv_id": doc.arxiv_id or "",
        }
        self.papers.upsert(
            ids=[doc.arxiv_id or doc.title],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )

    def store_implementation(
        self, module: CodeModule, code: str, understanding: PaperUnderstanding
    ) -> None:
        text = f"{module.name}: {module.description}\n{code[:500]}"
        embedding = self.embedder.encode([text])[0].tolist()
        self.implementations.upsert(
            ids=[f"{understanding.paper_title}::{module.id}"],
            embeddings=[embedding],
            documents=[text],
            metadatas={
                "module_type": module.id,
                "tech_stack": module.tech_stack,
                "paper": understanding.paper_title,
            },
        )

    def retrieve_similar_papers(
        self, query: str, n: int = 5, domain_filter: str = None
    ) -> list[dict]:
        embedding = self.embedder.encode([query])[0].tolist()
        where = {"domain": domain_filter} if domain_filter else None
        results = self.papers.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
        )
        return [
            {"text": d, "metadata": m}
            for d, m in zip(results["documents"][0], results["metadatas"][0])
        ]

    def retrieve_similar_implementations(
        self, module_description: str, tech_stack: str, n: int = 3
    ) -> list[str]:
        """Return past implementations similar to the requested module."""
        embedding = self.embedder.encode([module_description])[0].tolist()
        results = self.implementations.query(
            query_embeddings=[embedding],
            n_results=n,
            where={"tech_stack": tech_stack},
        )
        return results["documents"][0]
```

### 7.3 Wire knowledge base into generation

In `ModuleAgent._build_module_prompt()`, add a retrieval step:

```python
# After building context_block, before returning prompt:
if knowledge_base:
    similar = knowledge_base.retrieve_similar_implementations(
        module.description, plan.tech_stack, n=2
    )
    if similar:
        context_block += "\n\n# Similar implementations from knowledge base:\n"
        for s in similar:
            context_block += s[:1000] + "\n---\n"
```

Pass `knowledge_base: Optional[KnowledgeBase] = None` through `CodeOrchestrator`.

### 7.4 Wire into CLI

```
scholardevclaw kb stats                   # show stored papers + implementations
scholardevclaw kb search "attention mechanism"
scholardevclaw kb clear                   # wipe the local KB
```

### 7.5 Definition of Done — Phase 7

- [ ] After running the full pipeline on 2 papers, `kb stats` shows both.
- [ ] `kb search "layer normalization"` returns the relevant paper if it was
  ingested.
- [ ] A third paper in the same domain generates faster and with fewer healing
  rounds due to KB context (assert `generation_attempts` median is lower).
- [ ] KB persists across CLI invocations (kill process, restart, verify data
  still there).

---

## Phase 8 — Unified `from-paper` Command

**Goal:** Single command that runs the entire pipeline end-to-end. This is
the user-facing product.

### 8.1 Wire everything together in `cli.py`

```
scholardevclaw from-paper <pdf_or_doi_or_arxiv_id> \
    [--output-dir DIR]         # default: ./<project_name>/
    [--heal]                   # enable self-healing loop
    [--scaffold]               # generate API + demo + docs
    [--max-parallel N]         # default: 4
    [--model MODEL]            # default: claude-opus-4-5
    [--no-kb]                  # skip knowledge base lookup
    [--dry-run]                # show plan, don't generate
```

### 8.2 Pipeline implementation

```python
def cmd_from_paper(args):
    work_dir = Path(args.output_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    kb = None if args.no_kb else KnowledgeBase()

    # Phase 1: Ingest
    click.echo("[1/6] Ingesting paper...")
    fetcher = PaperFetcher()
    doc = fetcher.fetch_auto(args.source, work_dir / "paper")  # auto-detect type
    doc_path = work_dir / "paper_document.json"
    doc_path.write_text(json.dumps(doc.to_dict(), indent=2))

    # Phase 2: Understand
    click.echo("[2/6] Understanding paper...")
    agent = UnderstandingAgent(api_key=os.environ["ANTHROPIC_API_KEY"])
    understanding = agent.understand(doc)
    understanding_path = work_dir / "understanding.json"
    understanding_path.write_text(json.dumps(asdict(understanding), indent=2))

    # Phase 3: Plan
    click.echo("[3/6] Planning implementation...")
    planner = ImplementationPlanner(api_key=os.environ["ANTHROPIC_API_KEY"])
    plan = planner.plan(understanding, doc)
    plan_path = work_dir / "implementation_plan.json"
    plan_path.write_text(json.dumps(asdict(plan), indent=2))

    if args.dry_run:
        click.echo(f"\nDry run complete. Plan saved to {plan_path}")
        click.echo(f"Modules: {len(plan.modules)}, Stack: {plan.tech_stack}")
        return

    # Phase 4: Generate
    click.echo(f"[4/6] Generating {len(plan.modules)} modules...")
    project_dir = Path(args.output_dir) / plan.project_name
    orchestrator = CodeOrchestrator(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=args.model,
    )
    gen_result = orchestrator.generate_sync(
        plan, understanding, project_dir, args.max_parallel
    )
    click.echo(f"    Success rate: {gen_result.success_rate:.0%}")

    # Phase 5: Execute + heal
    if args.heal:
        click.echo("[5/6] Running tests + self-healing...")
        runner = SandboxRunner()
        healer = SelfHealingLoop(orchestrator, runner)
        gen_result = healer.heal(gen_result, plan, understanding)
    else:
        click.echo("[5/6] Running tests...")
        runner = SandboxRunner()

    exec_report = runner.run_tests(project_dir)
    scorer = ReproducibilityScorer(api_key=os.environ["ANTHROPIC_API_KEY"])
    repro = scorer.score(understanding, exec_report)
    click.echo(f"    Reproducibility: {repro.score:.0%} ({repro.verdict})")

    # Phase 6: Scaffold
    if args.scaffold:
        click.echo("[6/6] Generating product scaffold...")
        scaffolder = ProductScaffolder()
        scaffolder.scaffold(project_dir, plan, understanding, repro)

    # Knowledge base
    if kb:
        kb.store_paper(doc, understanding)
        for r in gen_result.module_results:
            if not r.final_errors:
                module = next(m for m in plan.modules if m.id == r.module_id)
                kb.store_implementation(module, r.code, understanding)

    # Final summary
    click.echo(f"\nDone. Project at: {project_dir.resolve()}")
    click.echo(f"  Modules: {len(plan.modules)}")
    click.echo(f"  Tests passed: {exec_report.tests_passed}")
    click.echo(f"  Reproducibility: {repro.score:.0%}")
    if args.scaffold:
        click.echo(f"  API: {project_dir}/api/main.py")
        click.echo(f"  Demo: {project_dir}/demo.py")
        click.echo(f"  Docs: {project_dir}/README.md")
```

### 8.3 Definition of Done — Phase 8 (system integration test)

Run the full pipeline on two papers:

**Test paper A** (simple): `arxiv:2003.00744` — EfficientDet

```bash
scholardevclaw from-paper arxiv:2003.00744 \
    --output-dir ./test_outputs/efficientdet \
    --heal --scaffold --dry-run
```

Expected: dry-run completes without error, plan shows ≥ 6 modules.

**Test paper B** (medium): `arxiv:2005.14165` — GPT-3 (architecture only, not training)

```bash
scholardevclaw from-paper arxiv:2005.14165 \
    --output-dir ./test_outputs/gpt3_arch \
    --heal --scaffold
```

Expected: `generation_report.json` has `success_rate > 0.7`. At least one
pytest test passes. README has non-empty reproducibility table.

---

## Non-Negotiable Code Standards

Every file produced in this upgrade must satisfy all of these:

**Typing:** Every function signature has type hints. Every class has typed
attributes. Use `from __future__ import annotations` at the top of every file.
Run `mypy --strict core/scholardevclaw/` — zero errors allowed.

**Error handling:** No bare `except:` or `except Exception:` without logging.
Use specific exception types. Network calls always have a timeout. LLM calls
always have a fallback message if they fail (never crash the CLI).

**Logging:** Use Python's `logging` module throughout. Log at DEBUG for LLM
prompts/responses, INFO for phase transitions, WARNING for degraded
situations, ERROR for failures. Never use `print()` except in CLI output
functions.

**Testing:** Every new module gets a corresponding `tests/test_<module>.py`.
Minimum coverage per module: 60%. Integration tests may use real API calls
only when marked `@pytest.mark.integration` and skipped by default.

**CLI UX:** Every long-running command shows a progress indicator. Use
`click.echo` for output, never `print`. Errors go to `stderr` via
`click.echo(..., err=True)`. Exit codes: 0=success, 1=partial failure,
2=complete failure, 3=configuration error.

**Secrets:** Never hardcode API keys. Always read from environment variables.
Raise `click.UsageError` with a helpful message if required env vars are missing.

---

## File Creation Checklist (in order)

```
core/scholardevclaw/ingestion/__init__.py
core/scholardevclaw/ingestion/models.py
core/scholardevclaw/ingestion/pdf_parser.py
core/scholardevclaw/ingestion/paper_fetcher.py
core/tests/test_ingestion.py

core/scholardevclaw/understanding/__init__.py
core/scholardevclaw/understanding/models.py
core/scholardevclaw/understanding/agent.py
core/scholardevclaw/understanding/graph.py
core/tests/test_understanding.py

core/scholardevclaw/planning/__init__.py
core/scholardevclaw/planning/models.py
core/scholardevclaw/planning/planner.py
core/tests/test_planning.py

core/scholardevclaw/generation/__init__.py
core/scholardevclaw/generation/models.py
core/scholardevclaw/generation/module_agent.py
core/scholardevclaw/generation/orchestrator.py
core/tests/test_generation.py

core/scholardevclaw/execution/__init__.py
core/scholardevclaw/execution/sandbox.py
core/scholardevclaw/execution/scorer.py
core/scholardevclaw/execution/healer.py
docker/sandbox.Dockerfile
core/tests/test_execution.py

core/scholardevclaw/product/__init__.py
core/scholardevclaw/product/scaffolder.py
core/scholardevclaw/product/templates/api_main.py.j2
core/scholardevclaw/product/templates/gradio_demo.py.j2
core/scholardevclaw/product/templates/README.md.j2
core/scholardevclaw/product/templates/Dockerfile.j2
core/tests/test_product.py

core/scholardevclaw/knowledge/__init__.py
core/scholardevclaw/knowledge/store.py
core/tests/test_knowledge.py

# Update existing
core/scholardevclaw/cli.py          (add all new commands + from-paper)
core/pyproject.toml                 (add dependency groups)
scripts/runbook.sh                  (add sandbox image build step)
```

---

## Environment Variables Reference

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional — defaults shown
SCHOLARDEVCLAW_LLM_MODEL=claude-opus-4-5
SCHOLARDEVCLAW_KB_DIR=~/.scholardevclaw/kb
SCHOLARDEVCLAW_SANDBOX_IMAGE=scholardevclaw-sandbox:latest
SCHOLARDEVCLAW_SANDBOX_TIMEOUT=300
SCHOLARDEVCLAW_SANDBOX_MEMORY_MB=4096
SCHOLARDEVCLAW_MAX_PARALLEL=4
SCHOLARDEVCLAW_LOG_LEVEL=INFO

# Existing (keep)
CORE_API_URL=http://localhost:8000
GITHUB_TOKEN=...
```

---

*End of UPGRADE.md — v3 target architecture for ScholarDevClaw.*
*Written for AI agent one-shot implementation.*
*Implement phases in order. Ship phase by phase. Do not skip.*
