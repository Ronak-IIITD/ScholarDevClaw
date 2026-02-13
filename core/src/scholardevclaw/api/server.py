from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json

from ..repo_intelligence.parser import PyTorchRepoParser
from ..research_intelligence.extractor import ResearchExtractor
from ..mapping.engine import MappingEngine
from ..patch_generation.generator import PatchGenerator
from ..validation.runner import ValidationRunner


app = FastAPI(
    title="ScholarDevClaw Core API",
    description="Autonomous ML Research Integration Engine - Core API",
    version="0.1.0",
)


class RepoAnalyzeRequest(BaseModel):
    repoPath: str


class ResearchExtractRequest(BaseModel):
    source: str
    sourceType: str = "pdf"


class MappingRequest(BaseModel):
    repoAnalysis: dict
    researchSpec: dict


class PatchGenerateRequest(BaseModel):
    mapping: dict


class ValidationRequest(BaseModel):
    patch: dict
    repoPath: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/repo/analyze")
async def analyze_repo(request: RepoAnalyzeRequest):
    try:
        repo_path = Path(request.repoPath)

        if not repo_path.exists():
            raise HTTPException(status_code=404, detail="Repository path not found")

        parser = PyTorchRepoParser(repo_path)
        repo_map = parser.parse()

        result = {
            "repoName": repo_map.repo_name,
            "architecture": {
                "models": [
                    {
                        "name": m.name,
                        "file": m.file,
                        "line": m.line,
                        "parent": m.parent,
                        "components": m.components,
                    }
                    for m in repo_map.models
                ],
                "trainingLoop": {
                    "file": repo_map.training_loop.file,
                    "line": repo_map.training_loop.line,
                    "optimizer": repo_map.training_loop.optimizer,
                    "lossFn": repo_map.training_loop.loss_fn,
                }
                if repo_map.training_loop
                else None,
            },
            "dependencies": {},
            "testSuite": {
                "runner": "pytest",
                "testFiles": repo_map.test_files,
            },
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/extract")
async def extract_research(request: ResearchExtractRequest):
    try:
        extractor = ResearchExtractor()
        result = extractor.extract(request.source, request.sourceType)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mapping/map")
async def map_architecture(request: MappingRequest):
    try:
        engine = MappingEngine(request.repoAnalysis, request.researchSpec)
        result = engine.map()

        return {
            "targets": [
                {
                    "file": t.file,
                    "line": t.line,
                    "currentCode": t.current_code,
                    "replacementRequired": t.replacement_required,
                }
                for t in result.targets
            ],
            "strategy": result.strategy,
            "confidence": result.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/patch/generate")
async def generate_patch(request: PatchGenerateRequest):
    try:
        generator = PatchGenerator(Path("."))
        patch = generator.generate(request.mapping)

        return {
            "newFiles": [{"path": f.path, "content": f.content} for f in patch.new_files],
            "transformations": [
                {
                    "file": t.file,
                    "original": t.original[:500],
                    "modified": t.modified[:500],
                    "changes": t.changes,
                }
                for t in patch.transformations
            ],
            "branchName": patch.branch_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validation/run")
async def run_validation(request: ValidationRequest):
    try:
        repo_path = Path(request.repoPath)

        if not repo_path.exists():
            raise HTTPException(status_code=404, detail="Repository path not found")

        runner = ValidationRunner(repo_path)
        result = runner.run(request.patch, request.repoPath)

        return {
            "passed": result.passed,
            "stage": result.stage,
            "baselineMetrics": {
                "loss": result.baseline_metrics.loss,
                "perplexity": result.baseline_metrics.perplexity,
                "tokensPerSecond": result.baseline_metrics.tokens_per_second,
                "memoryMb": result.baseline_metrics.memory_mb,
            }
            if result.baseline_metrics
            else None,
            "newMetrics": {
                "loss": result.new_metrics.loss,
                "perplexity": result.new_metrics.perplexity,
                "tokensPerSecond": result.new_metrics.tokens_per_second,
                "memoryMb": result.new_metrics.memory_mb,
            }
            if result.new_metrics
            else None,
            "comparison": result.comparison,
            "logs": result.logs,
            "error": result.error,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
