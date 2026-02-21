# Development Workflow

## Test-Driven Development (TDD)

Follow RED → GREEN → REFACTOR workflow:

1. **RED**: Write a failing test that defines desired behavior
2. **GREEN**: Implement minimal code to make the test pass
3. **REFACTOR**: Improve code quality while keeping tests green

## Package Management

- Use `uv` for all dependency and environment management
- `uv sync` — Install dependencies from pyproject.toml
- `uv run` — Run scripts with proper environment
- Modify `pyproject.toml`, never edit `requirements.txt` or `uv.lock` manually

## Python Version

- Python 3.10 only (pinned via `.python-version` and `pyproject.toml requires-python = "==3.10.*"`)
- Managed by `mise` for consistent Tcl/Tk support

## Commit Conventions

Use conventional commits for clear history and release automation:

- `feat:` — New features (minor version bump)
- `fix:` — Bug fixes (patch version bump)
- `feat!:` or `BREAKING CHANGE:` — Breaking changes (major version bump)
- `chore:`, `docs:`, `style:`, `refactor:` — No version bump
- All commits should be atomic and pass tests

## Task Runner

- Use `just` for project tasks — run `just --list` to see recipes
- Key recipes: `just setup`, `just start`, `just start-cpu`, `just clean`
- Recipes handle Tcl/Tk environment setup for GUI

## Running the Application

### With just (recommended)
```bash
just start            # Platform-default GPU
just start-cpu        # CPU only
just start-with rocm  # Specific provider
```

### With uv directly
```bash
uv run run.py --execution-provider coreml
```

### Headless mode (no Tcl/Tk needed)
```bash
uv run run.py -s source.jpg -t target.mp4 -o output.mp4
```

## Code Quality

- Prefer functional composition over class hierarchies
- Keep functions small and focused on a single responsibility
- Use pure functions without side effects when possible
- Data structures and functions over objects with methods
- YAGNI: Implement only what's immediately necessary
- DRY: Extract common logic, but avoid premature abstraction

## Error Handling

- Validate inputs early and fail immediately on invalid data
- Use "fail fast" principle — surface failures immediately
- Log errors comprehensively before failing
- Propagate errors up the stack to the appropriate handler

## Debugging

- Do not assume root cause without evidence
- Verify hypotheses against logs, timestamps, and context
- If diagnosis is corrected, fully re-investigate rather than patching
- Use systematic debugging approaches

### Crash Investigation (macOS)

Use the justfile debug recipes in order:

1. `just start-faulthandler` — `PYTHONFAULTHANDLER=1`; prints Python-level stack at SIGSEGV/SIGFPE
   before the interpreter dies. Best first step for any exit 139 or 133 crash.
2. `just crash-report` — parses the most recent `.ips` crash report from
   `~/Library/Logs/DiagnosticReports/python3.10-*.ips` and prints exception type, signal,
   and the crashing thread's native backtrace. Essential when Python-level stack is empty.
3. `just crash-list` — lists recent crash reports to pick a specific one.
4. `just start-lldb` — launches under lldb for interactive `bt`/`frame` inspection.
5. `just start-dyld-trace` — shows last shared library loaded before crash (useful for
   import-time segfaults in C extensions).

## Security

- Never commit API tokens or secrets
- Use environment variables for configuration
- Validate at system boundaries (user input, external APIs)
- Avoid command injection, XSS, SQL injection risks
