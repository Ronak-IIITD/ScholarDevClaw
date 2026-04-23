"""
Equation-to-Code Traceability Engine.

Maps paper equations to generated code lines, producing a traceability
report that shows exactly which equation each code block implements.

Features:
- Equation reference comments in generated code
- Traceability matrix (equation <-> code location)
- Markdown report generation for paper supplements
- Confidence scoring per mapping
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EquationReference:
    """A reference from code to a paper equation."""

    equation_id: str  # e.g. "eq_1", "eq_3_2"
    equation_latex: str  # raw LaTeX
    equation_description: str  # plain English description
    paper_section: str  # e.g. "§3.2"
    page: int = 0


@dataclass
class CodeMapping:
    """Maps a code location to a paper equation."""

    equation_ref: EquationReference
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str  # the relevant code lines
    confidence: float = 0.0  # 0.0 - 1.0
    mapping_type: str = "direct"  # "direct" | "partial" | "inspired_by"
    notes: str = ""


@dataclass
class TraceabilityReport:
    """Complete traceability report for a generated implementation."""

    paper_title: str
    paper_id: str  # arXiv ID or DOI
    implementation_dir: str
    total_equations: int
    mapped_equations: int
    unmapped_equations: list[str]
    mappings: list[CodeMapping]
    coverage_score: float = 0.0  # fraction of equations with code mappings
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Equation comment generator
# ---------------------------------------------------------------------------


def generate_equation_comment(
    eq_ref: EquationReference,
    mapping_type: str = "direct",
) -> str:
    """
    Generate a code comment that references a paper equation.

    Example output:
        # Equation 1 (§3.2): Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    lines = []

    # Header line with equation number and section
    header = f"# Equation {eq_ref.equation_id}"
    if eq_ref.paper_section:
        header += f" ({eq_ref.paper_section})"
    if eq_ref.equation_description:
        header += f": {eq_ref.equation_description}"
    lines.append(header)

    # LaTeX line (simplified for readability in code)
    if eq_ref.equation_latex:
        # Simplify LaTeX for code comments
        simplified = _simplify_latex_for_comment(eq_ref.equation_latex)
        lines.append(f"# {simplified}")

    if mapping_type == "partial":
        lines.append("# NOTE: Partial implementation of this equation")
    elif mapping_type == "inspired_by":
        lines.append("# NOTE: Inspired by (not exact reproduction of) this equation")

    return "\n".join(lines)


def _simplify_latex_for_comment(latex: str) -> str:
    """
    Simplify LaTeX notation for readable code comments.

    Converts common LaTeX to Unicode/ASCII approximations.
    """
    s = latex.strip()

    # Remove display-mode delimiters
    for delim in ["$$", "$", "\\[", "\\]", "\\begin{equation}", "\\end{equation}",
                   "\\begin{align}", "\\end{align}", "\\begin{gather}", "\\end{gather}"]:
        s = s.replace(delim, "")

    # Common substitutions
    replacements = {
        "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
        "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ", "\\eta": "η",
        "\\theta": "θ", "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν",
        "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ", "\\sigma": "σ",
        "\\tau": "τ", "\\phi": "φ", "\\chi": "χ", "\\psi": "ψ",
        "\\omega": "ω", "\\Omega": "Ω", "\\Sigma": "Σ", "\\Pi": "Π",
        "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
        "\\cdot": "·", "\\times": "×", "\\div": "÷",
        "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈",
        "\\infty": "∞", "\\partial": "∂", "\\nabla": "∇",
        "\\sum": "Σ", "\\prod": "Π", "\\int": "∫",
        "\\rightarrow": "→", "\\leftarrow": "←", "\\Rightarrow": "⇒",
        "\\in": "∈", "\\notin": "∉", "\\subset": "⊂",
        "\\forall": "∀", "\\exists": "∃",
        "\\mathbb{R}": "ℝ", "\\mathbb{N}": "ℕ", "\\mathbb{Z}": "ℤ",
        "\\text{softmax}": "softmax", "\\text{ReLU}": "ReLU",
        "\\text{LayerNorm}": "LayerNorm", "\\text{Attention}": "Attention",
        "\\mathrm{": "", "\\text{": "", "\\mathbf{": "", "\\mathcal{": "",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # Handle sqrt
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt\s+(\w)", r"√\1", s)

    # Handle fractions: \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)

    # Handle superscripts: x^{2} -> x²  (simple cases)
    sup_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
               "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
               "T": "ᵀ", "n": "ⁿ", "-1": "⁻¹"}
    for old, new in sup_map.items():
        s = s.replace(f"^{{{old}}}", new)
        s = s.replace(f"^{old}", new)

    # Handle subscripts: x_{i} -> x_i
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s)

    # Clean up remaining braces
    s = s.replace("{", "").replace("}", "")

    # Clean up whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ---------------------------------------------------------------------------
# Traceability builder
# ---------------------------------------------------------------------------


class TraceabilityBuilder:
    """
    Builds a traceability report by analyzing generated code against paper equations.
    """

    def __init__(
        self,
        paper_title: str,
        paper_id: str,
        equations: list[dict[str, Any]],
    ):
        self.paper_title = paper_title
        self.paper_id = paper_id
        self.equations = [
            EquationReference(
                equation_id=eq.get("id", f"eq_{i+1}"),
                equation_latex=eq.get("latex", ""),
                equation_description=eq.get("description", ""),
                paper_section=eq.get("section", ""),
                page=eq.get("page", 0),
            )
            for i, eq in enumerate(equations)
        ]
        self.mappings: list[CodeMapping] = []

    def add_mapping(
        self,
        equation_id: str,
        file_path: str,
        line_start: int,
        line_end: int,
        code_snippet: str,
        confidence: float = 0.8,
        mapping_type: str = "direct",
        notes: str = "",
    ) -> None:
        """Register a mapping between an equation and code location."""
        eq_ref = self._find_equation(equation_id)
        if eq_ref is None:
            logger.warning("Equation '%s' not found in paper", equation_id)
            return

        self.mappings.append(CodeMapping(
            equation_ref=eq_ref,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            code_snippet=code_snippet,
            confidence=confidence,
            mapping_type=mapping_type,
            notes=notes,
        ))

    def scan_code_for_references(self, project_dir: Path) -> None:
        """
        Scan generated code files for equation reference comments
        and auto-build mappings.

        Looks for patterns like:
            # Equation 1 (§3.2): description
            # Eq. 3: description
        """
        eq_pattern = re.compile(
            r"#\s*(?:Equation|Eq\.?)\s+(\w+(?:\.\w+)*)\s*(?:\(([^)]+)\))?\s*:?\s*(.*)",
            re.IGNORECASE,
        )

        for py_file in project_dir.rglob("*.py"):
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            rel_path = str(py_file.relative_to(project_dir))

            i = 0
            while i < len(lines):
                match = eq_pattern.match(lines[i].strip())
                if match:
                    eq_id = match.group(1)
                    section = match.group(2) or ""
                    description = match.group(3) or ""

                    # Find the code block following the comment
                    code_start = i + 1
                    # Skip additional comment lines
                    while code_start < len(lines) and lines[code_start].strip().startswith("#"):
                        code_start += 1

                    # Capture the next non-empty code block
                    code_end = code_start
                    while code_end < len(lines) and lines[code_end].strip():
                        code_end += 1

                    code_snippet = "\n".join(lines[code_start:code_end])

                    eq_ref = self._find_equation(eq_id)
                    if eq_ref:
                        self.mappings.append(CodeMapping(
                            equation_ref=eq_ref,
                            file_path=rel_path,
                            line_start=code_start + 1,  # 1-indexed
                            line_end=code_end,
                            code_snippet=code_snippet[:500],
                            confidence=0.9,  # High confidence — explicit comment
                            mapping_type="direct",
                        ))

                    i = code_end
                else:
                    i += 1

    def build_report(self, implementation_dir: str = "") -> TraceabilityReport:
        """Build the final traceability report."""
        from datetime import datetime

        mapped_eq_ids = {m.equation_ref.equation_id for m in self.mappings}
        all_eq_ids = {eq.equation_id for eq in self.equations}
        unmapped = sorted(all_eq_ids - mapped_eq_ids)

        coverage = len(mapped_eq_ids) / max(len(all_eq_ids), 1)

        return TraceabilityReport(
            paper_title=self.paper_title,
            paper_id=self.paper_id,
            implementation_dir=implementation_dir,
            total_equations=len(self.equations),
            mapped_equations=len(mapped_eq_ids),
            unmapped_equations=unmapped,
            mappings=self.mappings,
            coverage_score=coverage,
            generated_at=datetime.now().isoformat(),
        )

    def _find_equation(self, equation_id: str) -> EquationReference | None:
        """Find an equation by ID (flexible matching)."""
        # Exact match
        for eq in self.equations:
            if eq.equation_id == equation_id:
                return eq

        # Try numeric match (e.g., "1" matches "eq_1")
        for eq in self.equations:
            if eq.equation_id.replace("eq_", "") == equation_id:
                return eq
            if equation_id == eq.equation_id.split("_")[-1]:
                return eq

        return None


# ---------------------------------------------------------------------------
# Report exporter
# ---------------------------------------------------------------------------


def export_traceability_markdown(
    report: TraceabilityReport,
    output_path: Path | None = None,
) -> str:
    """
    Export a traceability report as a markdown document.

    Suitable for paper supplements, README sections, or documentation.
    """
    lines: list[str] = []

    lines.append(f"# Equation-to-Code Traceability Report")
    lines.append("")
    lines.append(f"**Paper:** {report.paper_title}")
    if report.paper_id:
        lines.append(f"**Paper ID:** {report.paper_id}")
    lines.append(f"**Generated:** {report.generated_at}")
    lines.append(f"**Coverage:** {report.coverage_score:.0%} "
                 f"({report.mapped_equations}/{report.total_equations} equations mapped)")
    lines.append("")

    # Coverage bar
    filled = int(report.coverage_score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"Coverage: [{bar}] {report.coverage_score:.0%}")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Traceability matrix
    lines.append("## Traceability Matrix")
    lines.append("")
    lines.append("| Equation | Section | File | Lines | Confidence | Type |")
    lines.append("|----------|---------|------|-------|------------|------|")

    for mapping in sorted(report.mappings, key=lambda m: m.equation_ref.equation_id):
        eq = mapping.equation_ref
        conf_bar = "●" * int(mapping.confidence * 5) + "○" * (5 - int(mapping.confidence * 5))
        lines.append(
            f"| {eq.equation_id} | {eq.paper_section} | "
            f"`{mapping.file_path}` | {mapping.line_start}-{mapping.line_end} | "
            f"{conf_bar} {mapping.confidence:.0%} | {mapping.mapping_type} |"
        )

    lines.append("")

    # Detailed mappings
    lines.append("## Detailed Mappings")
    lines.append("")

    for mapping in report.mappings:
        eq = mapping.equation_ref
        lines.append(f"### {eq.equation_id}: {eq.equation_description}")
        lines.append("")

        if eq.equation_latex:
            simplified = _simplify_latex_for_comment(eq.equation_latex)
            lines.append(f"**Equation:** `{simplified}`")
            lines.append("")

        lines.append(f"**File:** `{mapping.file_path}` (lines {mapping.line_start}–{mapping.line_end})")
        lines.append(f"**Confidence:** {mapping.confidence:.0%} | **Type:** {mapping.mapping_type}")
        lines.append("")

        if mapping.code_snippet:
            lines.append("```python")
            lines.append(mapping.code_snippet.strip())
            lines.append("```")
            lines.append("")

        if mapping.notes:
            lines.append(f"> {mapping.notes}")
            lines.append("")

    # Unmapped equations
    if report.unmapped_equations:
        lines.append("## ⚠️ Unmapped Equations")
        lines.append("")
        lines.append("The following equations do not have corresponding code mappings:")
        lines.append("")
        for eq_id in report.unmapped_equations:
            lines.append(f"- {eq_id}")
        lines.append("")

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Saved traceability report to %s", output_path)

    return content
