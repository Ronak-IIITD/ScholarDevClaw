# ScholarDevClaw - Quick Start Guide

This guide shows you how to use ScholarDevClaw to integrate ML research papers into your PyTorch repositories.

## Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw

# 2. Set up Python environment
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Verify installation
scholardevclaw --help
# or
python -m scholardevclaw.cli --help
```

## Quick Demo (with nanoGPT)

```bash
cd core
source .venv/bin/activate

# Run the demo to see it in action
python -m scholardevclaw.cli demo
```

## Using with Your Own Repository

### Step 1: Analyze Your Repository

```bash
# Analyze your PyTorch repo
python -m scholardevclaw.cli analyze /path/to/your/repo

# Or with JSON output
python -m scholardevclaw.cli analyze /path/to/your/repo --output-json
```

This will:

- Parse all Python files using AST
- Detect `nn.Module` subclasses
- Find training loops, optimizers, loss functions
- Identify test files

### Step 2: List Available Paper Specs

```bash
# See what papers are available
python -m scholardevclaw.cli specs --list
```

Currently available integrations:

- **RMSNorm** - Root Mean Square Layer Normalization (1910.07467)
- **SwiGLU** - Swish-Gated Linear Unit
- **FlashAttention** - IO-Aware Exact Attention (2205.14135)
- **RoPE** - Rotary Position Embedding (2104.09864)
- **GQA** - Grouped Query Attention (2305.13245)
- **Pre-LN** - Pre-Layer Normalization (2203.17056)
- **QKNorm** - Query-Key Normalization (2210.07440)
- **AdamW** - Decoupled Weight Decay (1711.05101)
- **FlashAttention-2** - Improved FlashAttention (2307.08691)
- **MoE** - Mixture of Experts (2401.04088)

### Step 3: Map Changes

```bash
# Map a paper spec to your repo
python -m scholardevclaw.cli map /path/to/your/repo rmsnorm
```

This will:

- Find where changes need to be made
- Check compatibility (tensor shapes, parameters)
- Calculate confidence score (0-100%)
- Show target locations

### Step 4: Generate Patch

```bash
# Generate the integration patch
python -m scholardevclaw.cli generate /path/to/your/repo rmsnorm --output-dir ./integration-patch
```

This will:

- Create new module files (e.g., `rmsnorm.py`)
- Generate code transformations
- Create a branch: `integration/<paper-name>`
- Save to specified directory

### Step 5: Validate

```bash
# Run tests and benchmarks
python -m scholardevclaw.cli validate /path/to/your/repo
```

This will:

- Run your existing test suite
- Execute lightweight training benchmark
- Compare metrics (loss, speed, memory)
- Generate comparison report

## Example: Integrating RMSNorm into Your Repo

```bash
# Full workflow
cd /your/project

# 1. See what models you have
python -m scholardevclaw.cli analyze .

# 2. Check if RMSNorm applies
python -m scholardevclaw.cli map . rmsnorm

# 3. Generate patch
python -m scholardevclaw.cli generate . rmsnorm --output-dir ./rmsnorm-integration

# 4. The patch creates:
#    - rmsnorm-integration/rmsnorm.py (new module)
#    - Branch: integration/rmsnorm
```

## Understanding the Output

### Confidence Scores

| Score   | Meaning                                     |
| ------- | ------------------------------------------- |
| 90-100% | High confidence - direct replacement        |
| 70-89%  | Good confidence - some manual review needed |
| 50-69%  | Moderate - manual inspection recommended    |
| <50%    | Low - significant changes required          |

### Patch Strategy Types

- **replace**: Direct substitution (e.g., LayerNorm → RMSNorm)
- **extend**: Add new module alongside existing
- **refactor**: Restructure needed

## Common Issues & Solutions

### "No models found"

Your repo might use a different pattern. Check:

- Are you using `nn.Module`?
- Is the model class in a `.py` file?

### "Targets: 0"

The spec might not match your code. Try:

- `python -m scholardevclaw.cli specs --list` to see available specs
- Map different paper that matches your architecture

### "Validation failed"

Check:

- Are there existing tests? (`pytest` should run)
- Does your training script work independently?

## Programmatic Usage (Python)

```python
from scholardevclaw.repo_intelligence.parser import PyTorchRepoParser
from scholardevclaw.research_intelligence.extractor import ResearchExtractor
from scholardevclaw.mapping.engine import MappingEngine
from scholardevclaw.patch_generation.generator import PatchGenerator

# Full pipeline
parser = PyTorchRepoParser("/path/to/repo")
repo_analysis = parser.parse()

extractor = ResearchExtractor()
spec = extractor.get_spec("rmsnorm")

engine = MappingEngine(repo_analysis, spec)
mapping = engine.map()

generator = PatchGenerator("/path/to/repo")
patch = generator.generate(mapping)

print(f"Branch: {patch.branch_name}")
print(f"Files: {[f.path for f in patch.new_files]}")
```

## Programmatic Usage (TypeScript/Agent)

```typescript
import { ScholarDevClawAgent } from "./agent/src/index.js";

const agent = new ScholarDevClawAgent({
  pythonCorePath: "../core/src",
  maxRetries: 2,
  benchmarkTimeout: 300,
});

const result = await agent.runIntegration(
  "/path/to/repo", // Repository
  "", // Paper source (optional)
  "rmsnorm", // Spec name
  "autonomous", // or 'step_approval'
);

console.log(result);
// Returns full integration context with all phase results
```

## Next Steps

1. **Review the patch** before applying
2. **Test locally** with `python -m scholardewclaw.cli validate`
3. **Create PR** using the generated branch
4. **Monitor benchmarks** in your CI

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           ScholarDevClaw                     │
├─────────────────────────────────────────────┤
│  Phase 1: Repo Intelligence (AST parsing)  │
│  Phase 2: Research Extraction (Paper specs) │
│  Phase 3: Mapping Engine (Compatibility)   │
│  Phase 4: Patch Generation (libcst)        │
│  Phase 5: Validation (Tests + Benchmark)  │
│  Phase 6: Report Generation                │
└─────────────────────────────────────────────┘
```

## Getting Help

- GitHub Issues: <https://github.com/Ronak-IIITD/ScholarDevClaw/issues>
- Check `AGENTS.md` for full architecture details
