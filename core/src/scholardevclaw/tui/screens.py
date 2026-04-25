"""Modal screens for the command-first TUI and paper-to-code workflow."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.generation.models import GenerationResult
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.llm.client import DEFAULT_MODELS
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding

from .clipboard import copy_to_clipboard

DEFAULT_OPENROUTER_MODEL = DEFAULT_MODELS[AuthProvider.OPENROUTER]
DEFAULT_TUI_PROVIDER_CHOICES: dict[str, AuthProvider] = {
    "anthropic": AuthProvider.ANTHROPIC,
    "openai": AuthProvider.OPENAI,
    "gemini": AuthProvider.GEMINI,
    "grok": AuthProvider.GROK,
    "moonshot": AuthProvider.MOONSHOT,
    "glm": AuthProvider.GLM,
    "minimax": AuthProvider.MINIMAX,
    "openrouter": AuthProvider.OPENROUTER,
    "ollama": AuthProvider.OLLAMA,
    "groq": AuthProvider.GROQ,
    "mistral": AuthProvider.MISTRAL,
    "deepseek": AuthProvider.DEEPSEEK,
    "cohere": AuthProvider.COHERE,
    "together": AuthProvider.TOGETHER,
    "fireworks": AuthProvider.FIREWORKS,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _meter(value: float, *, width: int = 24) -> str:
    clamped = _clamp01(value)
    filled = int(round(clamped * width))
    return f"{'█' * filled}{'░' * (width - filled)} {int(clamped * 100):>3}%"


WELCOME_TEXT = (
    "ScholarDevClaw\n\n"
    "Keyboard-first research-to-code shell.\n"
    "Paper workflow:\n"
    "  paper\n"
    "  paper arxiv:1706.03762\n"
    "  paper ./paper.pdf\n"
    "  Ctrl+P opens the paper workflow\n\n"
    "Repo workflow:\n"
    "  setup\n"
    "  /run analyze ./repo\n"
    "  /ask explain this repository\n"
    "  analyze ./repo\n"
    "  chat explain this repository\n"
    "  set mode search\n"
    "  :edit\n\n"
    "Press Enter or Esc to continue."
)


HELP_TEXT = (
    "Keys\n"
    "Tab autocomplete\n"
    "Up/Down history\n"
    "Ctrl+P open paper workflow\n"
    "Ctrl+I focus inspector\n"
    "Inspector: j/k move; Enter/Space events; r rerun; s show; e events\n"
    "Review mode: a/x/g set hunk, A/X/G set all, Enter/Space submit\n"
    "Ctrl+C cancel task or exit\n"
    "Ctrl+K clear output\n"
    "Enter execute\n"
    "Esc dismiss suggestions\n\n"
    "Setup\n"
    "setup\n"
    "set provider anthropic|openai|gemini|grok|moonshot|glm|minimax|openrouter|ollama\n"
    f"set model {DEFAULT_OPENROUTER_MODEL}\n\n"
    "Paper to Code\n"
    "paper [source]\n"
    "from-paper <source>\n"
    "  source: arXiv ID, DOI, URL, or local PDF\n\n"
    "Commands\n"
    "/ask <question>\n"
    "/run <action> [args...]\n"
    "  actions: analyze suggest search map generate validate integrate\n"
    "runs\n"
    "inspect\n"
    "run show <id>\n"
    "run events <id> [limit]\n"
    "run rerun <id>\n"
    "Backward compatible: analyze/map/generate/chat still work\n\n"
    "Modes\n"
    ":analyze\n"
    ":search\n"
    ":edit"
)


class WelcomeScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Dismiss"), ("enter", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    WelcomeScreen {
        align: left top;
        background: $background 80%;
    }

    WelcomeScreen > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(WELCOME_TEXT)


class HelpOverlay(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Dismiss"), ("enter", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    HelpOverlay {
        align: left top;
        background: $background 80%;
    }

    HelpOverlay > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(HELP_TEXT)


class ProviderSetupScreen(ModalScreen[dict[str, str] | None]):
    """Keyboard-first provider onboarding for the TUI shell."""

    BINDINGS = [
        ("ctrl+s", "submit_setup", "Save"),
        ("escape", "dismiss_skip", "Skip"),
    ]

    DEFAULT_CSS = """
    ProviderSetupScreen {
        align: center middle;
        background: $background 80%;
    }

    ProviderSetupScreen > Vertical {
        width: 60;
        height: 14;
        padding: 1 2;
        border: round $accent;
        background: $surface;
        overflow-y: auto;
    }

    ProviderSetupScreen Input {
        width: 100%;
        height: 3;
        border: solid $border;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    ProviderSetupScreen Input:focus {
        border: solid $accent;
    }

    #setup-hint {
        width: 100%;
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #setup-error {
        width: 100%;
        height: auto;
        color: $error;
    }
    """

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        model: str = "",
        has_saved_key: bool = False,
        supported_providers: dict[str, AuthProvider] | None = None,
        has_saved_key_by_provider: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self._supported_providers = dict(supported_providers or DEFAULT_TUI_PROVIDER_CHOICES)
        self._provider = (provider or "openrouter").strip().lower()
        if self._provider not in self._supported_providers:
            self._provider = next(iter(self._supported_providers), "openrouter")
        self._model = model
        self._has_saved_key = has_saved_key
        self._has_saved_key_by_provider = {
            str(key).strip().lower(): bool(value)
            for key, value in (has_saved_key_by_provider or {}).items()
        }

    def _provider_choices_text(self) -> str:
        return ", ".join(self._supported_providers)

    def _selected_provider(self) -> AuthProvider | None:
        return self._supported_providers.get(self._provider)

    def _selected_has_saved_key(self) -> bool:
        return self._has_saved_key_by_provider.get(self._provider, self._has_saved_key)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("LLM Setup")
            yield Static("", id="setup-hint")
            yield Input(
                value=self._provider,
                placeholder=f"Provider: {self._provider_choices_text()}",
                id="setup-provider",
            )
            yield Input(value=self._model, placeholder="Model ID", id="setup-model")
            yield Input(password=True, placeholder="API key (provider-specific)", id="setup-key")
            yield Static("", id="setup-error")

    def on_mount(self) -> None:
        self._refresh_hint()
        self.query_one("#setup-provider", Input).focus()

    def action_dismiss_skip(self) -> None:
        self.dismiss(None)

    def action_submit_setup(self) -> None:
        provider = self.query_one("#setup-provider", Input).value.strip().lower()
        model = self.query_one("#setup-model", Input).value.strip()
        api_key = self.query_one("#setup-key", Input).value.strip()
        error = self.query_one("#setup-error", Static)
        provider_choices = self._provider_choices_text()
        selected_provider = self._supported_providers.get(provider)

        if provider not in self._supported_providers:
            error.update(f"Error: provider must be one of: {provider_choices}")
            return
        if not model:
            error.update("Error: model is required")
            return
        if (
            selected_provider is not None
            and selected_provider.requires_api_key
            and not api_key
            and not self._has_saved_key_by_provider.get(provider, self._has_saved_key)
        ):
            error.update(f"Error: {selected_provider.display_name} requires an API key")
            return

        self.dismiss(
            {
                "provider": provider,
                "model": model,
                "api_key": api_key,
            }
        )

    @on(Input.Changed, "#setup-provider")
    def on_provider_changed(self, event: Input.Changed) -> None:
        self._provider = event.value.strip().lower()
        self._refresh_hint()

    @on(Input.Submitted, "#setup-provider")
    def on_provider_submitted(self) -> None:
        self.query_one("#setup-model", Input).focus()

    @on(Input.Submitted, "#setup-model")
    def on_model_submitted(self) -> None:
        selected_provider = self._selected_provider()
        if selected_provider is not None and not selected_provider.requires_api_key:
            self.action_submit_setup()
            return
        self.query_one("#setup-key", Input).focus()

    @on(Input.Submitted, "#setup-key")
    def on_key_submitted(self) -> None:
        self.action_submit_setup()

    def _refresh_hint(self) -> None:
        hint = self.query_one("#setup-hint", Static)
        key_input = self.query_one("#setup-key", Input)
        provider = self._selected_provider()
        if provider is None:
            hint.update(
                "Provider -> choose one of the supported providers\n"
                f"Supported -> {self._provider_choices_text()}\n"
                "Model -> enter a provider model id\n"
                "Key -> provider-specific\n"
                "Save -> Ctrl+S or Enter"
            )
            key_input.placeholder = "API key (provider-specific)"
            return

        model_example = DEFAULT_MODELS.get(provider, "provider-default-model")
        if not provider.requires_api_key:
            hint.update(
                f"Provider -> {provider.display_name}\n"
                f"Model -> for example `{model_example}`\n"
                "Key -> not required\n"
                "Save -> Ctrl+S or Enter"
            )
            key_input.placeholder = "No key required for this provider"
            return

        reuse = (
            "leave key blank to reuse saved key"
            if self._selected_has_saved_key()
            else f"paste your {provider.display_name} key"
        )
        hint.update(
            f"Provider -> {provider.display_name}\n"
            f"Model -> for example `{model_example}`\n"
            f"Key format -> {provider.key_format_hint}\n"
            f"Key -> {reuse}\n"
            "Save -> Ctrl+S or Enter"
        )
        key_input.placeholder = f"{provider.display_name} API key"


class CommandPalette(ModalScreen[str | None]):
    """Thin command chooser for keyboard fallback."""

    BINDINGS = [
        ("escape", "dismiss_none", "Dismiss"),
        ("enter", "run_selected", "Run"),
        ("down", "select_next", "Next"),
        ("up", "select_prev", "Prev"),
    ]

    PALETTE_COMMANDS = [
        "paper",
        "paper arxiv:1706.03762",
        "paper ./paper.pdf",
        "from-paper arxiv:1706.03762",
        "setup",
        "/ask explain this repository",
        "/run analyze ./repo",
        "/run generate ./repo rmsnorm",
        "analyze ./repo",
        "suggest ./repo",
        "chat hello",
        "search layer normalization",
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "validate ./repo",
        "runs",
        "inspect",
        "run show 1",
        "run events 1",
        "run rerun 1",
        "set provider openrouter",
        "set provider ollama",
        f"set model {DEFAULT_OPENROUTER_MODEL}",
        "set mode analyze",
        "set dir ./repo",
        ":analyze",
        ":search",
        ":edit",
    ]

    DEFAULT_CSS = """
    CommandPalette {
        align: left top;
        background: $background 80%;
    }

    CommandPalette > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }

    CommandPalette Input {
        width: 100%;
        height: 1;
        border: none;
        padding: 0;
    }

    CommandPalette .palette-line {
        width: 100%;
        height: 1;
        color: $text-muted;
    }

    CommandPalette .palette-line.-selected {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._selected_index = 0
        self._filtered_commands = list(self.PALETTE_COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Commands")
            yield Input(placeholder="Filter commands", id="palette-input")
            for command in self.PALETTE_COMMANDS[:3]:
                yield Static(command, classes="palette-line")

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()
        self._refresh()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_select_next(self) -> None:
        if self._filtered_commands:
            self._selected_index = (self._selected_index + 1) % len(self._filtered_commands)
            self._refresh()

    def action_select_prev(self) -> None:
        if self._filtered_commands:
            self._selected_index = (self._selected_index - 1) % len(self._filtered_commands)
            self._refresh()

    def action_run_selected(self) -> None:
        if not self._filtered_commands:
            self.dismiss(None)
            return
        self.dismiss(self._filtered_commands[self._selected_index])

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        if not query:
            self._filtered_commands = list(self.PALETTE_COMMANDS)
        else:
            self._filtered_commands = [cmd for cmd in self.PALETTE_COMMANDS if query in cmd.lower()]
        self._selected_index = 0
        self._refresh()

    def _refresh(self) -> None:
        lines = list(self.query(".palette-line"))
        for line in lines:
            line.remove()
        if not self._filtered_commands:
            self.mount(Static("No matches", classes="palette-line"))
            return
        for index, command in enumerate(self._filtered_commands[:3]):
            classes = "palette-line"
            if index == self._selected_index:
                classes += " -selected"
            self.mount(Static(command, classes=classes))


class PaperIngestionScreen(ModalScreen[dict[str, str] | None]):
    """Paper-to-code entry: ingest paper source and preview extracted structures."""

    BINDINGS = [("escape", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    PaperIngestionScreen {
        align: center middle;
        background: $background 88%;
    }

    PaperIngestionScreen > Vertical {
        width: 120;
        height: 36;
        border: round $accent;
        background: #111827;
        padding: 1 2;
    }

    #paper-source-input {
        width: 100%;
        margin: 1 0;
    }

    #paper-ingestion-main {
        height: 1fr;
    }

    #paper-info,
    #paper-preview {
        width: 1fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #paper-status {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    """

    def __init__(
        self,
        *,
        paper_source: str = "",
        paper_document: PaperDocument | None = None,
    ) -> None:
        super().__init__()
        self._paper_source = paper_source
        self._paper_document = paper_document
        self._status = "Idle. Enter an arXiv ID, DOI, URL, or local PDF path."
        self._algorithms: list[str] = []
        self._equations: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Ingestion")
            yield Input(
                value=self._paper_source,
                placeholder="Examples: 2406.12345, 10.1145/1234567, https://arxiv.org/abs/..., ./paper.pdf",
                id="paper-source-input",
            )
            with Horizontal(id="paper-ingestion-main"):
                yield Static("", id="paper-info")
                yield Static("", id="paper-preview")
            yield Static(self._status, id="paper-status")

    def on_mount(self) -> None:
        self.query_one("#paper-source-input", Input).focus()
        self._render_paper_info()
        self._render_preview()

    @on(Input.Submitted, "#paper-source-input")
    def on_source_submitted(self, event: Input.Submitted) -> None:
        self._paper_source = event.value.strip()
        if not self._paper_source:
            self.set_status("Paper source is required.")
            return
        self.set_status(f"Parsing paper source: {self._paper_source}")
        self.dismiss({"source": self._paper_source})

    def set_status(self, status: str) -> None:
        self._status = status.strip() or "Working..."
        self.query_one("#paper-status", Static).update(self._status)

    def set_paper_document(self, paper_document: PaperDocument) -> None:
        self._paper_document = paper_document
        self._algorithms = [algo.name for algo in paper_document.algorithms if algo.name.strip()]
        self._equations = [eq.latex for eq in paper_document.equations if eq.latex.strip()]
        self._render_paper_info()
        self._render_preview()
        self.set_status("Paper metadata extracted.")

    def add_algorithm(self, name: str) -> None:
        value = name.strip()
        if value:
            self._algorithms.append(value)
            self._render_preview()

    def add_equation(self, latex: str) -> None:
        value = latex.strip()
        if value:
            self._equations.append(value)
            self._render_preview()

    def _render_paper_info(self) -> None:
        paper = self._paper_document
        if paper is None:
            info = (
                "Paper Metadata\n"
                "──────────────\n"
                "Title: (waiting)\n"
                "Authors: (waiting)\n"
                "Year: (waiting)\n"
                "arXiv: (waiting)\n"
                "DOI: (waiting)"
            )
        else:
            authors = ", ".join(paper.authors[:5]) if paper.authors else "Unknown"
            if len(paper.authors) > 5:
                authors += ", …"
            info = (
                "Paper Metadata\n"
                "──────────────\n"
                f"Title: {paper.title or 'Unknown'}\n"
                f"Authors: {authors}\n"
                f"Year: {paper.year if paper.year is not None else 'Unknown'}\n"
                f"arXiv: {paper.arxiv_id or '-'}\n"
                f"DOI: {paper.doi or '-'}\n"
                f"Sections: {len(paper.sections)}\n"
                f"Figures: {len(paper.figures)}"
            )
        self.query_one("#paper-info", Static).update(info)

    def _render_preview(self) -> None:
        algo_lines = [f"  • {name}" for name in self._algorithms[-8:]] or ["  • (none yet)"]
        equation_lines = [f"  • {eq[:90]}" for eq in self._equations[-8:]] or ["  • (none yet)"]
        preview = (
            "Detected Algorithms + Equations\n"
            "─────────────────────────────\n"
            "Algorithms\n" + "\n".join(algo_lines) + "\n\nEquations\n" + "\n".join(equation_lines)
        )
        self.query_one("#paper-preview", Static).update(preview)


class UnderstandingScreen(ModalScreen[dict[str, str] | None]):
    """Paper-to-code review: inspect parsed understanding and confirm quality."""

    BINDINGS = [("escape", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    UnderstandingScreen {
        align: center middle;
        background: $background 88%;
    }

    UnderstandingScreen > Vertical {
        width: 122;
        height: 38;
        border: round $accent;
        background: #0f172a;
        padding: 1 2;
    }

    #understanding-main {
        height: 1fr;
    }

    #understanding-graph {
        width: 2fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #understanding-sidebar {
        width: 1fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #understanding-actions {
        margin: 1 0 0 0;
        height: auto;
    }
    """

    def __init__(self, *, understanding: PaperUnderstanding | None = None) -> None:
        super().__init__()
        self._understanding = understanding

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Understanding")
            with Horizontal(id="understanding-main"):
                yield Static("", id="understanding-graph")
                yield Static("", id="understanding-sidebar")
            with Horizontal(id="understanding-actions"):
                yield Button("Looks good → proceed", id="understanding-proceed", variant="success")
                yield Button("Edit understanding", id="understanding-edit", variant="warning")

    def on_mount(self) -> None:
        self._refresh_all()

    def set_understanding(self, understanding: PaperUnderstanding) -> None:
        self._understanding = understanding
        self._refresh_all()

    def _refresh_all(self) -> None:
        understanding = self._understanding
        if understanding is None:
            self.query_one("#understanding-graph", Static).update(
                "Concept Graph (ASCII)\n────────────────────\n(waiting for understanding data)"
            )
            self.query_one("#understanding-sidebar", Static).update(
                "Contributions\n  • (none)\n\nRequirements\n  • (none)\n\n"
                "Complexity: unknown\nEstimated hours: 0\n"
                f"Confidence: {_meter(0.0)}"
            )
            return

        nodes = {node.id: node.label or node.id for node in understanding.concept_nodes}
        edge_lines: list[str] = []
        for edge in understanding.concept_edges[:30]:
            src = nodes.get(edge.source_id, edge.source_id)
            dst = nodes.get(edge.target_id, edge.target_id)
            edge_lines.append(f"{src} --{edge.relation or 'rel'}--> {dst}")
        if not edge_lines:
            edge_lines = ["(no graph edges extracted)"]
        graph = "Concept Graph (ASCII)\n────────────────────\n" + "\n".join(edge_lines)

        contribution_lines = [
            f"  • {item.claim}" for item in understanding.contributions[:6] if item.claim.strip()
        ] or ["  • (none)"]
        requirement_lines = [
            f"  • {item.name} ({item.requirement_type})"
            for item in understanding.requirements[:8]
            if item.name.strip()
        ] or ["  • (none)"]

        sidebar = (
            "Contributions\n"
            + "\n".join(contribution_lines)
            + "\n\nRequirements\n"
            + "\n".join(requirement_lines)
            + "\n\n"
            + f"Complexity: {understanding.complexity}\n"
            + f"Estimated hours: {understanding.estimated_impl_hours}\n"
            + f"Confidence: {_meter(understanding.confidence)}"
        )

        self.query_one("#understanding-graph", Static).update(graph)
        self.query_one("#understanding-sidebar", Static).update(sidebar)

    @on(Button.Pressed, "#understanding-proceed")
    def on_understanding_proceed(self) -> None:
        self.dismiss({"decision": "proceed"})

    @on(Button.Pressed, "#understanding-edit")
    def on_understanding_edit(self) -> None:
        self.dismiss({"decision": "edit"})


class PlanningScreen(ModalScreen[dict[str, Any] | None]):
    """Paper-to-code review: inspect implementation plan and pass approval gate."""

    BINDINGS = [("escape", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    PlanningScreen {
        align: center middle;
        background: $background 88%;
    }

    PlanningScreen > Vertical {
        width: 122;
        height: 38;
        border: round $accent;
        background: #0b1220;
        padding: 1 2;
    }

    #planning-main {
        height: 1fr;
    }

    #planning-details,
    #planning-estimates,
    #planning-tech {
        width: 1fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #planning-gate {
        margin: 1 0 0 0;
        color: $warning;
    }

    #planning-actions {
        margin: 1 0 0 0;
    }
    """

    def __init__(self, *, plan: ImplementationPlan | None = None) -> None:
        super().__init__()
        self._plan = plan
        self.approved: bool | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Planning")
            with Horizontal(id="planning-main"):
                yield Static("", id="planning-details")
                yield Static("", id="planning-estimates")
                yield Static("", id="planning-tech")
            yield Static(
                "Approval gate: confirm this plan before generation starts.", id="planning-gate"
            )
            with Horizontal(id="planning-actions"):
                yield Button("Approve", id="planning-approve", variant="success")
                yield Button("Reject", id="planning-reject", variant="error")

    def on_mount(self) -> None:
        self._refresh_all()

    def set_plan(self, plan: ImplementationPlan) -> None:
        self._plan = plan
        self._refresh_all()

    def _refresh_all(self) -> None:
        plan = self._plan
        if plan is None:
            self.query_one("#planning-details", Static).update(
                "Module Dependency Graph\n───────────────────────\n(waiting for plan)"
            )
            self.query_one("#planning-estimates", Static).update(
                "Per-Module Estimates\n────────────────────\n(waiting for plan)"
            )
            self.query_one("#planning-tech", Static).update(
                "Tech Stack\n──────────\n(waiting for plan)"
            )
            return

        graph_lines = ["Module Dependency Graph", "───────────────────────"]
        if not plan.modules:
            graph_lines.append("(no modules)")
        else:
            for module in sorted(plan.modules, key=lambda item: item.priority):
                dep = ", ".join(module.depends_on) if module.depends_on else "root"
                graph_lines.append(f"├─ {module.id or module.name}")
                graph_lines.append(f"│  └─ depends_on: {dep}")
        self.query_one("#planning-details", Static).update("\n".join(graph_lines))

        estimate_lines = ["Per-Module Estimates", "────────────────────"]
        for module in sorted(plan.modules, key=lambda item: item.priority):
            est_lines = max(0, module.estimated_lines)
            est_minutes = max(15, est_lines * 2)
            est_tokens = est_lines * 6
            estimate_lines.append(
                f"• {module.id or module.name}: {est_lines} lines, {est_minutes} min, ~{est_tokens} tokens"
            )
        estimate_lines.append("")
        estimate_lines.append(f"Total estimated lines: {plan.estimated_total_lines}")
        self.query_one("#planning-estimates", Static).update("\n".join(estimate_lines))

        tech = (
            "Tech Stack\n"
            "──────────\n"
            f"Language: {plan.target_language}\n"
            f"Stack: {plan.tech_stack or 'unspecified'}\n\n"
            "Justification\n"
            "• Matches target language from planning stage\n"
            "• Keeps dependencies scoped to listed modules\n"
            "• Optimizes for implementation speed and maintainability"
        )
        self.query_one("#planning-tech", Static).update(tech)

    @on(Button.Pressed, "#planning-approve")
    def on_approve(self) -> None:
        self.approved = True
        self.query_one("#planning-gate", Static).update(
            "Approval gate: approved. Generation can proceed."
        )
        self.dismiss({"approved": True, "decision": "approve"})

    @on(Button.Pressed, "#planning-reject")
    def on_reject(self) -> None:
        self.approved = False
        self.query_one("#planning-gate", Static).update(
            "Approval gate: rejected. Adjust plan before generation."
        )
        self.dismiss({"approved": False, "decision": "reject"})


class GenerationScreen(ModalScreen[None]):
    """Paper-to-code generation monitor with live logs and progress."""

    BINDINGS = [
        ("escape", "dismiss", "Dismiss"),
        ("left", "prev_module", "Prev Module"),
        ("right", "next_module", "Next Module"),
    ]

    DEFAULT_CSS = """
    GenerationScreen {
        align: center middle;
        background: $background 88%;
    }

    GenerationScreen > Vertical {
        width: 124;
        height: 40;
        border: round $accent;
        background: #020617;
        padding: 1 2;
    }

    #generation-tabs,
    #generation-overall,
    #generation-module-progress,
    #generation-alerts,
    #generation-log {
        width: 100%;
        border: solid $border;
        padding: 1;
    }

    #generation-log {
        height: 1fr;
        overflow-y: auto;
    }

    #generation-actions {
        margin: 1 0 0 0;
    }
    """

    def __init__(
        self,
        *,
        module_ids: list[str] | None = None,
        generation_result: GenerationResult | None = None,
    ) -> None:
        super().__init__()
        self._module_order = list(module_ids or [])
        self._current_module_index = 0
        self._module_logs: dict[str, list[str]] = {
            module_id: [] for module_id in self._module_order
        }
        self._module_progress: dict[str, float] = {
            module_id: 0.0 for module_id in self._module_order
        }
        self._alerts: list[str] = []
        self._overall_progress = 0.0
        self._token_usage = 0
        self._cancelled = False
        if generation_result is not None:
            self.set_generation_result(generation_result)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Generation")
            yield Static("", id="generation-tabs")
            yield Static("", id="generation-overall")
            yield Static("", id="generation-module-progress")
            yield Static("", id="generation-alerts")
            yield Static("", id="generation-log")
            with Horizontal(id="generation-actions"):
                yield Button("◀ Prev", id="generation-prev")
                yield Button("Next ▶", id="generation-next")
                yield Button("Cancel", id="generation-cancel", variant="error")

    def on_mount(self) -> None:
        self._refresh_all()

    def action_prev_module(self) -> None:
        if not self._module_order:
            return
        self._current_module_index = (self._current_module_index - 1) % len(self._module_order)
        self._refresh_tabs_and_log()

    def action_next_module(self) -> None:
        if not self._module_order:
            return
        self._current_module_index = (self._current_module_index + 1) % len(self._module_order)
        self._refresh_tabs_and_log()

    @on(Button.Pressed, "#generation-prev")
    def on_prev_pressed(self) -> None:
        self.action_prev_module()

    @on(Button.Pressed, "#generation-next")
    def on_next_pressed(self) -> None:
        self.action_next_module()

    @on(Button.Pressed, "#generation-cancel")
    def on_cancel_pressed(self) -> None:
        self._cancelled = True
        self._alerts.append("Generation cancelled by user")
        self._refresh_alerts()

    def set_generation_result(self, result: GenerationResult) -> None:
        module_ids = [item.module_id for item in result.module_results if item.module_id.strip()]
        for module_id in module_ids:
            if module_id not in self._module_order:
                self._module_order.append(module_id)
        for row in result.module_results:
            self._module_logs.setdefault(row.module_id, []).append(
                f"Generated: {row.file_path} / {row.test_file_path}"
            )
            self._module_progress[row.module_id] = 1.0 if not row.final_errors else 0.75
            self._token_usage += max(0, row.tokens_used)
            if row.final_errors:
                for error in row.final_errors:
                    self._alerts.append(f"{row.module_id}: {error}")
        self._overall_progress = _clamp01(result.success_rate)
        self._refresh_all()

    def append_module_log(self, module_id: str, line: str) -> None:
        module = module_id.strip() or "default"
        if module not in self._module_order:
            self._module_order.append(module)
            self._module_logs[module] = []
            self._module_progress.setdefault(module, 0.0)
        self._module_logs.setdefault(module, []).append(line.rstrip())
        self._refresh_tabs_and_log()

    def set_module_progress(self, module_id: str, progress: float) -> None:
        module = module_id.strip() or "default"
        if module not in self._module_order:
            self._module_order.append(module)
            self._module_logs.setdefault(module, [])
        self._module_progress[module] = _clamp01(progress)
        if self._module_progress:
            self._overall_progress = _clamp01(
                sum(self._module_progress.values()) / max(1, len(self._module_progress))
            )
        self._refresh_progress()

    def add_syntax_error(self, module_id: str, error_message: str) -> None:
        self._alerts.append(f"Syntax error [{module_id}]: {error_message.strip()}")
        self._refresh_alerts()

    def increment_token_usage(self, token_count: int) -> None:
        self._token_usage += max(0, token_count)
        self._refresh_progress()

    def _refresh_all(self) -> None:
        self._refresh_tabs_and_log()
        self._refresh_progress()
        self._refresh_alerts()

    def _refresh_tabs_and_log(self) -> None:
        if not self._module_order:
            self.query_one("#generation-tabs", Static).update("Modules: (none)")
            self.query_one("#generation-log", Static).update(
                "Live module log\n───────────────\n(waiting)"
            )
            return

        labels: list[str] = []
        for index, module in enumerate(self._module_order):
            if index == self._current_module_index:
                labels.append(f"[{module}]")
            else:
                labels.append(module)
        tabs = "Modules: " + " | ".join(labels)
        self.query_one("#generation-tabs", Static).update(tabs)

        selected = self._module_order[self._current_module_index]
        lines = self._module_logs.get(selected, [])[-120:]
        body = "\n".join(lines) if lines else "(no logs yet)"
        self.query_one("#generation-log", Static).update(
            f"Live log · {selected}\n────────────────\n{body}"
        )

    def _refresh_progress(self) -> None:
        overall = (
            f"Overall progress: {_meter(self._overall_progress)}\n"
            f"Token usage: {self._token_usage:,}"
        )
        self.query_one("#generation-overall", Static).update(overall)

        lines = ["Per-module progress", "──────────────────"]
        for module in self._module_order:
            lines.append(f"• {module}: {_meter(self._module_progress.get(module, 0.0), width=16)}")
        if len(lines) == 2:
            lines.append("(none)")
        self.query_one("#generation-module-progress", Static).update("\n".join(lines))

    def _refresh_alerts(self) -> None:
        if not self._alerts:
            text = "Syntax alerts\n─────────────\n(none)"
        else:
            items = [f"• {item}" for item in self._alerts[-8:]]
            text = "Syntax alerts\n─────────────\n" + "\n".join(items)
        self.query_one("#generation-alerts", Static).update(text)


class ExecutionScreen(ModalScreen[None]):
    """Paper-to-code execution, healing, and reproducibility monitoring."""

    BINDINGS = [("escape", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    ExecutionScreen {
        align: center middle;
        background: $background 88%;
    }

    ExecutionScreen > Vertical {
        width: 122;
        height: 38;
        border: round $accent;
        background: #020617;
        padding: 1 2;
    }

    #execution-healing {
        color: $warning;
        margin: 0 0 1 0;
    }

    #execution-main {
        height: 1fr;
    }

    #execution-output,
    #execution-status {
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #execution-output {
        width: 2fr;
    }

    #execution-status {
        width: 1fr;
    }
    """

    def __init__(self, *, report: ReproducibilityReport | None = None) -> None:
        super().__init__()
        self._pytest_lines: list[str] = []
        self._test_status: dict[str, str] = {}
        self._healing_round = 0
        self._healing_total: int | None = None
        self._repro_score = 0.0
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Execution")
            yield Static("Healing round: 0", id="execution-healing")
            with Horizontal(id="execution-main"):
                yield Static("", id="execution-output")
                yield Static("", id="execution-status")

    def on_mount(self) -> None:
        self._refresh_all()

    def append_pytest_output(self, line: str) -> None:
        self._pytest_lines.append(line.rstrip())
        self._refresh_output()

    def set_test_status(self, test_name: str, *, passed: bool) -> None:
        self._test_status[test_name] = "PASS" if passed else "FAIL"
        self._refresh_status()

    def set_healing_round(self, round_index: int, *, total_rounds: int | None = None) -> None:
        self._healing_round = max(0, round_index)
        self._healing_total = total_rounds
        if total_rounds is None:
            text = f"Healing round: {self._healing_round}"
        else:
            text = f"Healing round: {self._healing_round}/{max(1, total_rounds)}"
        self.query_one("#execution-healing", Static).update(text)

    def update_reproducibility_score(self, score: float) -> None:
        self._repro_score = _clamp01(score)
        self._refresh_status()

    def set_reproducibility_report(self, report: ReproducibilityReport) -> None:
        self._report = report
        self._repro_score = _clamp01(report.score)
        self._refresh_status()

    def _refresh_all(self) -> None:
        self._refresh_output()
        self._refresh_status()

    def _refresh_output(self) -> None:
        text = (
            "\n".join(self._pytest_lines[-220:])
            if self._pytest_lines
            else "(pytest output pending)"
        )
        self.query_one("#execution-output", Static).update(
            "Live pytest output\n──────────────────\n" + text
        )

    def _refresh_status(self) -> None:
        tests = ["Per-test status", "──────────────"]
        if not self._test_status:
            tests.append("(none)")
        else:
            for name, status in list(self._test_status.items())[-24:]:
                tests.append(f"• {status:<4} {name}")

        tests.append("")
        tests.append("Reproducibility")
        tests.append("──────────────")
        tests.append(_meter(self._repro_score, width=20))
        if self._report is not None:
            tests.append(f"Verdict: {self._report.verdict}")
            tests.append(f"Metrics: {len(self._report.achieved_metrics)} achieved")
        self.query_one("#execution-status", Static).update("\n".join(tests))


class ProductScreen(ModalScreen[None]):
    """Paper-to-code output browser with generated artifacts and quick actions."""

    BINDINGS = [
        ("escape", "dismiss", "Dismiss"),
        ("up", "select_prev", "Prev File"),
        ("down", "select_next", "Next File"),
        ("enter", "open_editor", "Open in editor"),
    ]

    DEFAULT_CSS = """
    ProductScreen {
        align: center middle;
        background: $background 88%;
    }

    ProductScreen > Vertical {
        width: 126;
        height: 42;
        border: round $accent;
        background: #020617;
        padding: 1 2;
    }

    #product-main {
        height: 1fr;
    }

    #product-tree {
        width: 1fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #product-preview {
        width: 2fr;
        height: 1fr;
        border: solid $border;
        padding: 1;
        overflow-y: auto;
    }

    #product-actions {
        margin: 1 0 0 0;
    }

    #product-status {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    """

    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        install_command: str = "pip install -e .",
    ) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._install_command = install_command
        self._files: list[Path] = []
        self._selected_index = 0
        self._status = "Ready"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Paper to Code · Product")
            with Horizontal(id="product-main"):
                yield Static("", id="product-tree")
                yield Static("", id="product-preview")
            with Horizontal(id="product-actions"):
                yield Button("◀", id="product-prev")
                yield Button("▶", id="product-next")
                yield Button("Open in editor", id="product-open", variant="primary")
                yield Button("Copy install command", id="product-copy", variant="success")
            yield Static("", id="product-status")

    def on_mount(self) -> None:
        self._reload_files()
        self._refresh_all()

    def set_output_dir(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._reload_files()
        self._refresh_all()

    def set_install_command(self, command: str) -> None:
        self._install_command = command.strip() or "pip install -e ."
        self._status = "Install command updated"
        self._refresh_status()

    def action_select_prev(self) -> None:
        if not self._files:
            return
        self._selected_index = (self._selected_index - 1) % len(self._files)
        self._refresh_tree_and_preview()

    def action_select_next(self) -> None:
        if not self._files:
            return
        self._selected_index = (self._selected_index + 1) % len(self._files)
        self._refresh_tree_and_preview()

    def action_open_editor(self) -> None:
        selected = self._selected_file()
        if selected is None:
            self._status = "No file selected"
            self._refresh_status()
            return
        editor = (os.environ.get("EDITOR") or "vi").strip() or "vi"
        try:
            editor_cmd = shlex.split(editor)
            subprocess.Popen([*editor_cmd, str(selected)])
            self._status = f"Opened in editor: {selected.name}"
        except Exception as exc:
            self._status = f"Failed to open editor: {exc}"
        self._refresh_status()

    @on(Button.Pressed, "#product-prev")
    def on_prev_pressed(self) -> None:
        self.action_select_prev()

    @on(Button.Pressed, "#product-next")
    def on_next_pressed(self) -> None:
        self.action_select_next()

    @on(Button.Pressed, "#product-open")
    def on_open_pressed(self) -> None:
        self.action_open_editor()

    @on(Button.Pressed, "#product-copy")
    def on_copy_pressed(self) -> None:
        copied = copy_to_clipboard(self._install_command)
        self._status = (
            f"Copied install command: {self._install_command}"
            if copied
            else "Could not copy command to clipboard"
        )
        self._refresh_status()

    def _reload_files(self) -> None:
        self._files = []
        if self._output_dir is None:
            self._status = "No generated output directory provided"
            return
        base = self._output_dir.expanduser().resolve()
        if not base.exists() or not base.is_dir():
            self._status = f"Output directory not found: {base}"
            return

        self._files = sorted(
            [path for path in base.rglob("*") if path.is_file()],
            key=lambda path: str(path).lower(),
        )[:500]
        self._selected_index = 0
        self._status = f"Loaded {len(self._files)} files"

    def _selected_file(self) -> Path | None:
        if not self._files:
            return None
        return self._files[self._selected_index]

    def _refresh_all(self) -> None:
        self._refresh_tree_and_preview()
        self._refresh_status()

    def _refresh_tree_and_preview(self) -> None:
        selected = self._selected_file()
        if self._output_dir is None:
            self.query_one("#product-tree", Static).update(
                "File tree\n────────\n(no output directory)"
            )
            self.query_one("#product-preview", Static).update("Preview\n───────\n(none)")
            return

        base = self._output_dir.expanduser().resolve()
        if not self._files:
            self.query_one("#product-tree", Static).update("File tree\n────────\n(no files)")
            self.query_one("#product-preview", Static).update("Preview\n───────\n(none)")
            return

        tree_lines = ["Generated file tree", "───────────────────"]
        for index, path in enumerate(self._files[:220]):
            marker = "▶" if index == self._selected_index else " "
            rel = path.relative_to(base)
            tree_lines.append(f"{marker} {rel}")
        self.query_one("#product-tree", Static).update("\n".join(tree_lines))

        preview_lines = ["File preview", "────────────"]
        if selected is None:
            preview_lines.append("(no file selected)")
        else:
            preview_lines.append(str(selected.relative_to(base)))
            preview_lines.append("")
            try:
                content = selected.read_text(encoding="utf-8", errors="replace")
                preview_lines.extend(content.splitlines()[:160])
            except Exception as exc:
                preview_lines.append(f"Could not read file: {exc}")
        self.query_one("#product-preview", Static).update("\n".join(preview_lines))

    def _refresh_status(self) -> None:
        self.query_one("#product-status", Static).update(self._status)
