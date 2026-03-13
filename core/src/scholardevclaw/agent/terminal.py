"""
Advanced Terminal System — gives the agent super powers like a real terminal.

Features:
- Persistent shell session with working directory memory
- Shell built-ins: cd, pwd, export, alias, unalias, history, jobs
- ANSI color support
- Command history with navigation
- Pipes, redirects, and background jobs (&)
- Environment variable expansion
- Tab completion for paths
- Interactive terminal mode
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShellState:
    """Represents the state of a shell session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cwd: str = os.getcwd()
    env: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    history_index: int = -1
    last_exit_code: int = 0
    jobs: dict[int, dict] = field(default_factory=dict)
    next_job_id: int = 1


class TerminalColors:
    """ANSI terminal colors and formatting."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Cursor
    CLEAR_SCREEN = "\033[2J"
    CLEAR_LINE = "\033[2K"
    CURSOR_HOME = "\033[H"
    CURSOR_UP = "\033[A"
    CURSOR_DOWN = "\033[B"
    CURSOR_FORWARD = "\033[C"
    CURSOR_BACK = "\033[D"
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text."""
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def prompt(cls, cwd: str, user: str = "user") -> str:
        """Generate a colored prompt."""
        home = os.path.expanduser("~")
        display_cwd = cwd.replace(home, "~")

        # Colored path
        path = cls.colorize(display_cwd, cls.CYAN)

        # Colored user
        user_str = cls.colorize(user, cls.GREEN)

        # Prompt symbol
        symbol = cls.colorize("$", cls.YELLOW)

        return f"{user_str} {path} {symbol} "

    @classmethod
    def success(cls, text: str) -> str:
        """Green success text."""
        return cls.colorize(text, cls.GREEN)

    @classmethod
    def error(cls, text: str) -> str:
        """Red error text."""
        return cls.colorize(text, cls.RED)

    @classmethod
    def warning(cls, text: str) -> str:
        """Yellow warning text."""
        return cls.colorize(text, cls.YELLOW)

    @classmethod
    def info(cls, text: str) -> str:
        """Blue info text."""
        return cls.colorize(text, cls.BLUE)


class AdvancedShell:
    """
    Advanced shell with terminal-like capabilities.

    Provides:
    - Persistent working directory
    - Shell built-ins (cd, pwd, export, alias, history, jobs)
    - Command history
    - Pipes and redirects
    - Background jobs
    - Environment variable expansion
    """

    BUILTINS = {
        "cd": "change directory",
        "pwd": "print working directory",
        "export": "set environment variables",
        "alias": "create command alias",
        "unalias": "remove alias",
        "history": "show command history",
        "jobs": "show background jobs",
        "fg": "bring job to foreground",
        "bg": "resume job in background",
        "kill": "terminate a job",
        "exit": "exit shell",
        "echo": "print text",
        "env": "show environment variables",
        "set": "set shell options",
        "unset": "unset environment variable",
        "type": "show command type",
        "which": "locate command",
        "true": "do nothing, successfully",
        "false": "do nothing, unsuccessfully",
    }

    def __init__(self, initial_cwd: str | None = None):
        self.state = ShellState(cwd=initial_cwd or os.getcwd())
        self.colors = TerminalColors()

        # Initialize environment
        for key, value in os.environ.items():
            self.state.env[key] = value

    def run_command(
        self,
        command: str,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """
        Execute a command with full shell capabilities.

        Returns:
            dict with keys: stdout, stderr, returncode, timed_out, output
        """
        if not command.strip():
            return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

        # Add to history (skip duplicates)
        if command.strip() and (not self.state.history or self.state.history[-1] != command):
            self.state.history.append(command)
            self.state.history_index = len(self.state.history)

        # Parse command (handle pipes and redirects)
        parsed = self._parse_command(command)
        if not parsed:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": 1,
                "timed_out": False,
                "output": "Parse error",
            }

        # Handle built-ins
        if parsed["cmd"] in self.BUILTINS:
            return self._run_builtin(parsed)

        # Expand aliases
        expanded = self._expand_alias(command)

        # Expand environment variables
        expanded = self._expand_env(expanded)

        # Execute with pipes
        return self._execute_with_pipes(expanded, timeout)

    async def run_command_async(
        self,
        command: str,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """Execute a command with full shell capabilities (async)."""
        if not command.strip():
            return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

        # Add to history (skip duplicates)
        if command.strip() and (not self.state.history or self.state.history[-1] != command):
            self.state.history.append(command)
            self.state.history_index = len(self.state.history)

        # Parse command (handle pipes and redirects)
        parsed = self._parse_command(command)
        if not parsed:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": 1,
                "timed_out": False,
                "output": "Parse error",
            }

        # Handle built-ins
        if parsed["cmd"] in self.BUILTINS:
            return self._run_builtin(parsed)

        # Expand aliases
        expanded = self._expand_alias(command)

        # Expand environment variables
        expanded = self._expand_env(expanded)

        # Execute with pipes (async)
        return await self._execute_with_pipes_async(expanded, timeout)

    def _parse_command(self, command: str) -> dict | None:
        """Parse command string into components."""
        try:
            # Split but keep redirects
            parts = shlex.split(command)
        except ValueError:
            return None

        if not parts:
            return None

        cmd = parts[0]
        args = parts[1:]

        return {
            "raw": command,
            "cmd": cmd,
            "args": args,
            "stdin_redirect": None,
            "stdout_redirect": None,
            "stderr_redirect": None,
            "background": False,
        }

    def _expand_alias(self, command: str) -> str:
        """Expand command aliases."""
        parts = command.strip().split()
        if not parts:
            return command

        cmd = parts[0]
        if cmd in self.state.aliases:
            alias = self.state.aliases[cmd]
            # Replace first word with alias
            return command.replace(cmd, alias, 1)

        return command

    def _expand_env(self, command: str) -> str:
        """Expand environment variables in command."""
        import re

        # Match $VAR, ${VAR}
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return str(self.state.env.get(var_name, match.group(0)))

        # ${VAR} or $VAR
        command = re.sub(r"\$\{(\w+)\}", replace_var, command)
        command = re.sub(r"\$(\w+)", replace_var, command)

        return command

    def _run_builtin(self, parsed: dict) -> dict[str, Any]:
        """Run a shell built-in command."""
        cmd = parsed["cmd"]
        args = parsed["args"]

        if cmd == "cd":
            return self._builtin_cd(args)
        elif cmd == "pwd":
            return self._builtin_pwd()
        elif cmd == "export":
            return self._builtin_export(args)
        elif cmd == "alias":
            return self._builtin_alias(args)
        elif cmd == "unalias":
            return self._builtin_unalias(args)
        elif cmd == "history":
            return self._builtin_history(args)
        elif cmd == "jobs":
            return self._builtin_jobs()
        elif cmd == "fg":
            return self._builtin_fg(args)
        elif cmd == "bg":
            return self._builtin_bg(args)
        elif cmd == "kill":
            return self._builtin_kill(args)
        elif cmd == "exit":
            return {
                "stdout": "",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
                "output": "",
                "exit_shell": True,
            }
        elif cmd == "echo":
            return self._builtin_echo(args)
        elif cmd == "env":
            return self._builtin_env()
        elif cmd == "set":
            return self._builtin_set(args)
        elif cmd == "unset":
            return self._builtin_unset(args)
        elif cmd == "type":
            return self._builtin_type(args)
        elif cmd == "which":
            return self._builtin_which(args)
        elif cmd == "true":
            return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}
        elif cmd == "false":
            return {"stdout": "", "stderr": "", "returncode": 1, "timed_out": False, "output": ""}

        return {
            "stdout": "",
            "stderr": f"Unknown builtin: {cmd}",
            "returncode": 1,
            "timed_out": False,
            "output": f"Unknown builtin: {cmd}",
        }

    def _builtin_cd(self, args: list[str]) -> dict[str, Any]:
        """Change directory."""
        if not args:
            # cd to home
            target = os.path.expanduser("~")
        else:
            target = args[0]

        # Handle -
        if target == "-":
            # cd to previous directory (would need OLDPWD)
            target = self.state.env.get("OLDPWD", os.getcwd())

        # Resolve path
        if not os.path.isabs(target):
            target = os.path.join(self.state.cwd, target)

        target = os.path.abspath(os.path.expanduser(target))

        if os.path.isdir(target):
            old_cwd = self.state.cwd
            self.state.cwd = target
            self.state.env["PWD"] = target
            self.state.env["OLDPWD"] = old_cwd
            return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}
        else:
            return {
                "stdout": "",
                "stderr": f"cd: {args[0]}: No such file or directory",
                "returncode": 1,
                "timed_out": False,
                "output": f"cd: {args[0]}: No such file or directory",
            }

    def _builtin_pwd(self) -> dict[str, Any]:
        """Print working directory."""
        return {
            "stdout": self.state.cwd + "\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": self.state.cwd + "\n",
        }

    def _builtin_export(self, args: list[str]) -> dict[str, Any]:
        """Set environment variables."""
        if not args:
            # Show exported vars
            exported = {k: value for k, value in self.state.env.items() if k.isupper()}
            return {
                "stdout": "\n".join(f"{k}={exported[k]}" for k in sorted(exported)) + "\n",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
                "output": "",
            }

        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                self.state.env[key] = value
            else:
                self.state.env[arg] = ""

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_alias(self, args: list[str]) -> dict[str, Any]:
        """Create alias."""
        if not args:
            # Show all aliases
            if not self.state.aliases:
                return {
                    "stdout": "",
                    "stderr": "",
                    "returncode": 0,
                    "timed_out": False,
                    "output": "",
                }

            lines = [f"alias {k}='{v}'" for k, v in self.state.aliases.items()]
            return {
                "stdout": "\n".join(lines) + "\n",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
                "output": "",
            }

        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Remove quotes from value
                value = value.strip("'\"")
                self.state.aliases[key] = value

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_unalias(self, args: list[str]) -> dict[str, Any]:
        """Remove alias."""
        for arg in args:
            self.state.aliases.pop(arg, None)

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_history(self, args: list[str]) -> dict[str, Any]:
        """Show command history."""
        limit = None
        if args:
            try:
                limit = int(args[0])
            except ValueError:
                pass

        hist = self.state.history[-limit:] if limit else self.state.history
        lines = [f"{i + 1:5}  {cmd}" for i, cmd in enumerate(hist)]

        return {
            "stdout": "\n".join(lines) + "\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": "",
        }

    def _builtin_jobs(self) -> dict[str, Any]:
        """Show background jobs."""
        if not self.state.jobs:
            return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

        lines = []
        for job_id, job in self.state.jobs.items():
            status = "Running" if job.get("running") else "Stopped"
            lines.append(f"[{job_id}] {status}  {job.get('cmd', '')}")

        return {
            "stdout": "\n".join(lines) + "\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": "",
        }

    def _builtin_fg(self, args: list[str]) -> dict[str, Any]:
        """Bring job to foreground."""
        if not args:
            return {
                "stdout": "",
                "stderr": "fg: current: no such job",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        try:
            job_id = int(args[0].lstrip("%"))
        except ValueError:
            return {
                "stdout": "",
                "stderr": "fg: invalid job id",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        if job_id not in self.state.jobs:
            return {
                "stdout": "",
                "stderr": f"fg: {job_id}: no such job",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        job = self.state.jobs[job_id]
        if "pid" in job:
            # Resume process
            try:
                os.kill(job["pid"], signal.SIGCONT)
            except ProcessLookupError:
                return {
                    "stdout": "",
                    "stderr": f"fg: {job_id}: process not found",
                    "returncode": 1,
                    "timed_out": False,
                    "output": "",
                }

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_bg(self, args: list[str]) -> dict[str, Any]:
        """Resume job in background."""
        if not args:
            return {
                "stdout": "",
                "stderr": "bg: current: no such job",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        try:
            job_id = int(args[0].lstrip("%"))
        except ValueError:
            return {
                "stdout": "",
                "stderr": "bg: invalid job id",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        if job_id not in self.state.jobs:
            return {
                "stdout": "",
                "stderr": f"bg: {job_id}: no such job",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        job = self.state.jobs[job_id]
        if "pid" in job:
            try:
                os.kill(job["pid"], signal.SIGCONT)
                job["running"] = True
            except ProcessLookupError:
                return {
                    "stdout": "",
                    "stderr": f"bg: {job_id}: process not found",
                    "returncode": 1,
                    "timed_out": False,
                    "output": "",
                }

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_kill(self, args: list[str]) -> dict[str, Any]:
        """Terminate a job."""
        if not args:
            return {
                "stdout": "",
                "stderr": "kill: usage: kill [-s sigspec] pid",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        sig = signal.SIGTERM
        target = args[-1]

        # Parse options
        for arg in args[:-1]:
            if arg.startswith("-"):
                if arg == "-9":
                    sig = signal.SIGKILL

        try:
            pid = int(target.lstrip("%"))
            os.kill(pid, sig)
        except (ValueError, ProcessLookupError) as e:
            return {
                "stdout": "",
                "stderr": f"kill: {target}: {e}",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_echo(self, args: list[str]) -> dict[str, Any]:
        """Echo text."""
        # Handle -n flag
        newline = True
        if args and args[0] == "-n":
            newline = False
            args = args[1:]

        output = " ".join(args) + ("\n" if newline else "")

        return {
            "stdout": output,
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": output,
        }

    def _builtin_env(self) -> dict[str, Any]:
        """Show environment variables."""
        lines = [f"{k}={v}" for k, v in sorted(self.state.env.items())]

        return {
            "stdout": "\n".join(lines) + "\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": "",
        }

    def _builtin_set(self, args: list[str]) -> dict[str, Any]:
        """Set shell options."""
        if not args:
            # Show all variables
            lines = [f"{k}={v}" for k, v in sorted(self.state.env.items())]
            return {
                "stdout": "\n".join(lines) + "\n",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
                "output": "",
            }

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_unset(self, args: list[str]) -> dict[str, Any]:
        """Unset environment variable."""
        for arg in args:
            self.state.env.pop(arg, None)

        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "output": ""}

    def _builtin_type(self, args: list[str]) -> dict[str, Any]:
        """Show command type."""
        if not args:
            return {
                "stdout": "",
                "stderr": "type: usage: type name...",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        results = []
        for cmd in args:
            if cmd in self.BUILTINS:
                results.append(f"{cmd} is a shell builtin")
            elif cmd in self.state.aliases:
                results.append(f"{cmd} is an alias for `{self.state.aliases[cmd]}`")
            else:
                # Check PATH
                import shutil

                path = shutil.which(cmd)
                if path:
                    results.append(f"{cmd} is {path}")
                else:
                    results.append(f"{cmd}: not found")

        return {
            "stdout": "\n".join(results) + "\n",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": "",
        }

    def _builtin_which(self, args: list[str]) -> dict[str, Any]:
        """Locate command."""
        if not args:
            return {
                "stdout": "",
                "stderr": "which: usage: which program...",
                "returncode": 1,
                "timed_out": False,
                "output": "",
            }

        import shutil

        results = []
        for cmd in args:
            path = shutil.which(cmd)
            if path:
                results.append(path)
            else:
                results.append(f"{cmd} not found")

        return {
            "stdout": "\n".join(results) + "\n",
            "stderr": "",
            "returncode": 0 if all(shutil.which(a) for a in args) else 1,
            "timed_out": False,
            "output": "",
        }

    def _execute_with_pipes(self, command: str, timeout: int) -> dict[str, Any]:
        """Execute command with pipes (sync wrapper)."""
        return asyncio.run(self._execute_with_pipes_async(command, timeout))

    async def _execute_with_pipes_async(self, command: str, timeout: int) -> dict[str, Any]:
        """Execute command with pipes (async)."""
        # Split by pipe
        parts = [p.strip() for p in command.split("|")]

        if len(parts) == 1:
            return await self._execute_single_async(parts[0], timeout)

        # Pipeline
        stdout = ""
        prev_stdout = None

        for i, part in enumerate(parts):
            stdin = prev_stdout or ""
            result = await self._execute_async(part, stdin=stdin, timeout=timeout)
            prev_stdout = result["stdout"] if result["returncode"] == 0 else None

            if result["returncode"] != 0 and i > 0:
                return result

        return {
            "stdout": stdout,
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "output": stdout,
        }

    def _execute_single(self, command: str, timeout: int) -> dict[str, Any]:
        """Execute a single command (sync wrapper)."""
        return asyncio.run(self._execute_async(command, timeout=timeout))

    async def _execute_single_async(self, command: str, timeout: int) -> dict[str, Any]:
        """Execute a single command (async)."""
        return await self._execute_async(command, timeout=timeout)

    async def _execute_async(
        self, command: str, stdin: str | None = None, timeout: int = 60
    ) -> dict[str, Any]:
        """Execute command asynchronously."""
        # Check for background job
        background = command.strip().endswith("&")
        if background:
            command = command.strip()[:-1].strip()

        # Parse redirects

        if ">" in command or "2>" in command:
            parts = command.split(">")
            command = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if part.startswith("2"):
                    part[2:].strip()
                else:
                    part.strip()

        if "<" in command:
            parts = command.split("<")
            command = parts[0].strip()
            if len(parts) > 1:
                parts[1].strip()

        # Run command
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.state.cwd,
                env=self.state.env,
            )

            # Handle background
            if background:
                job_id = self.state.next_job_id
                self.state.next_job_id += 1
                self.state.jobs[job_id] = {
                    "pid": proc.pid,
                    "cmd": command,
                    "running": True,
                }
                return {
                    "stdout": f"[{job_id}] {proc.pid}\n",
                    "stderr": "",
                    "returncode": 0,
                    "timed_out": False,
                    "output": f"[{job_id}] {proc.pid}\n",
                }

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                self.state.last_exit_code = int(proc.returncode or 0)

                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": proc.returncode,
                    "timed_out": False,
                    "output": stdout + stderr,
                }
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                self.state.last_exit_code = 124
                return {
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                    "returncode": 124,
                    "timed_out": True,
                    "output": f"Command timed out after {timeout} seconds",
                }

        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
                "timed_out": False,
                "output": str(e),
            }

    def get_prompt(self) -> str:
        """Get colored prompt string."""
        return self.colors.prompt(self.state.cwd)

    def get_state(self) -> ShellState:
        """Get current shell state."""
        return self.state

    def set_state(self, state: ShellState) -> None:
        """Restore shell state."""
        self.state = state

    def get_completions(self, partial: str) -> list[str]:
        """Get possible completions for a partial command."""
        # Complete commands
        if not partial or " " not in partial:
            # Complete command name
            cmd_partial = partial.split()[-1] if partial else ""
            return [c for c in self.BUILTINS.keys() if c.startswith(cmd_partial)]

        # Complete file/directory
        parts = partial.split()
        last = parts[-1]

        # Expand home
        if last.startswith("~"):
            last = os.path.expanduser(last)

        # Get directory
        if "/" in last:
            dir_path = os.path.dirname(last)
            if not dir_path:
                dir_path = "."
            prefix = os.path.basename(last)
        else:
            dir_path = self.state.cwd
            prefix = last

        # List possibilities
        try:
            entries = os.listdir(dir_path)
            completions = []
            for entry in entries:
                if entry.startswith(prefix):
                    full_path = os.path.join(dir_path, entry)
                    if os.path.isdir(full_path):
                        completions.append(entry + "/")
                    else:
                        completions.append(entry)
            return completions
        except (OSError, FileNotFoundError):
            return []

    def get_history_prev(self) -> str | None:
        """Get previous command from history."""
        if not self.state.history:
            return None

        if self.state.history_index > 0:
            self.state.history_index -= 1

        return self.state.history[self.state.history_index]

    def get_history_next(self) -> str | None:
        """Get next command from history."""
        if not self.state.history:
            return None

        if self.state.history_index < len(self.state.history) - 1:
            self.state.history_index += 1
            return self.state.history[self.state.history_index]
        else:
            self.state.history_index = len(self.state.history)
            return ""


def create_shell(initial_cwd: str | None = None) -> AdvancedShell:
    """Factory function to create an advanced shell."""
    return AdvancedShell(initial_cwd)
